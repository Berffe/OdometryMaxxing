/****************************************************************************
 *
 *   BEE_LAND Gazebo-native truth system
 *
 ****************************************************************************/

#include "BeeLandingTruth.hpp"

#include <gz/common/Console.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

using namespace custom;

GZ_ADD_PLUGIN(
	BeeLandingTruth,
	gz::sim::System,
	gz::sim::ISystemConfigure,
	gz::sim::ISystemPreUpdate,
	gz::sim::ISystemPostUpdate
)

GZ_ADD_PLUGIN_ALIAS(BeeLandingTruth, "custom::BeeLandingTruth")

namespace
{
constexpr double kTruthSchemaVersion = 1.0;
constexpr double kTimeEpsilon = 1e-9;

bool contains(const std::string &text, const std::string &token)
{
	return !token.empty() && text.find(token) != std::string::npos;
}

float asFloat(double value)
{
	return static_cast<float>(value);
}

} // namespace

void BeeLandingTruth::Configure(const gz::sim::Entity &entity,
		const std::shared_ptr<const sdf::Element> &sdf,
		gz::sim::EntityComponentManager &ecm,
		gz::sim::EventManager &eventMgr)
{
	(void)eventMgr;
	_worldEntity = entity;
	gz::sim::World world(entity);
	if (const auto name = world.Name(ecm); name.has_value()) {
		_worldName = *name;
	}

	_droneModelName = readSdfString(sdf, "drone_model", _droneModelName);
	_platformModelName = readSdfString(sdf, "platform_model", _platformModelName);
	_droneBaseLinkName = readSdfString(sdf, "drone_base_link", _droneBaseLinkName);
	_cameraLinkName = readSdfString(sdf, "camera_link", _cameraLinkName);
	_platformLinkName = readSdfString(sdf, "platform_link", _platformLinkName);
	_gearLinkName = readSdfString(sdf, "gear_link", _gearLinkName);
	_leftSensorName = readSdfString(sdf, "left_contact_sensor", _leftSensorName);
	_rightSensorName = readSdfString(sdf, "right_contact_sensor", _rightSensorName);
	_contactTargetSubstring = readSdfString(sdf, "contact_target_substring", _contactTargetSubstring);
	_truthTopic = readSdfString(sdf, "truth_topic", _truthTopic);

	_leftPadPointBase = readSdfVector3(sdf, "left_pad_point_base", _leftPadPointBase);
	_rightPadPointBase = readSdfVector3(sdf, "right_pad_point_base", _rightPadPointBase);
	_skidHalfLengthM = readSdfDouble(sdf, "skid_half_length_m", _skidHalfLengthM);
	_skidHalfWidthM = readSdfDouble(sdf, "skid_half_width_m", _skidHalfWidthM);
	_platformTopOffsetM = readSdfDouble(sdf, "platform_top_offset_m", _platformTopOffsetM);
	_publishRateHz = readSdfDouble(sdf, "publish_rate_hz", _publishRateHz);
	_contactConfirmationSec = readSdfDouble(sdf, "contact_confirmation_sec", _contactConfirmationSec);
	_contactStaleTimeoutSec = readSdfDouble(sdf, "contact_stale_timeout_sec", _contactStaleTimeoutSec);
	_divergenceMinDistanceM = readSdfDouble(sdf, "divergence_min_distance_m", _divergenceMinDistanceM);
	_minimumViewAlignment = readSdfDouble(sdf, "minimum_view_alignment", _minimumViewAlignment);

	const std::string defaultLeftTopic =
		"/world/" + _worldName + "/model/" + _droneModelName +
		"/link/" + _gearLinkName + "/sensor/" + _leftSensorName + "/contact";
	const std::string defaultRightTopic =
		"/world/" + _worldName + "/model/" + _droneModelName +
		"/link/" + _gearLinkName + "/sensor/" + _rightSensorName + "/contact";

	_leftContactTopic = readSdfString(sdf, "left_contact_topic", defaultLeftTopic);
	_rightContactTopic = readSdfString(sdf, "right_contact_topic", defaultRightTopic);

	_truthPublisher = _transportNode.Advertise<gz::msgs::Float_V>(_truthTopic);
	const bool leftSubscribed = _transportNode.Subscribe(
		_leftContactTopic, &BeeLandingTruth::onLeftContacts, this);
	const bool rightSubscribed = _transportNode.Subscribe(
		_rightContactTopic, &BeeLandingTruth::onRightContacts, this);

	gzmsg << "[BeeLandingTruth] publishing fixed-layout truth on [" << _truthTopic << "]\n";
	gzmsg << "[BeeLandingTruth] left contact topic:  [" << _leftContactTopic
	      << "] subscribed=" << leftSubscribed << "\n";
	gzmsg << "[BeeLandingTruth] right contact topic: [" << _rightContactTopic
	      << "] subscribed=" << rightSubscribed << "\n";

	// The drone is spawned later by PX4, so failure here is expected. PreUpdate
	// retries until every required entity exists.
	resolveEntities(ecm);
}

void BeeLandingTruth::PreUpdate(const gz::sim::UpdateInfo &info,
		gz::sim::EntityComponentManager &ecm)
{
	if (info.paused) {
		return;
	}

	if (_entitiesReady) {
		const bool stillValid =
			_droneBaseLink.Valid(ecm) && _cameraLink.Valid(ecm) && _platformLink.Valid(ecm);
		if (!stillValid) {
			clearEntityHandles();
		}
	}

	if (!_entitiesReady) {
		resolveEntities(ecm);
	}
}

void BeeLandingTruth::PostUpdate(const gz::sim::UpdateInfo &info,
		const gz::sim::EntityComponentManager &ecm)
{
	if (info.paused || !_entitiesReady) {
		return;
	}

	const double simTimeSec = std::chrono::duration<double>(info.simTime).count();
	if (finite(_lastObservedSimTimeSec) && simTimeSec + kTimeEpsilon < _lastObservedSimTimeSec) {
		// World reset / log replay jump. Event latches belong to one run only.
		resetRuntimeState();
	}
	_lastObservedSimTimeSec = simTimeSec;

	publishTruth(info, ecm, false);
}

bool BeeLandingTruth::resolveEntities(gz::sim::EntityComponentManager &ecm)
{
	const gz::sim::Entity droneEntity = ecm.EntityByComponents(
		gz::sim::components::Model(),
		gz::sim::components::Name(_droneModelName),
		gz::sim::components::ParentEntity(_worldEntity));
	const gz::sim::Entity platformEntity = ecm.EntityByComponents(
		gz::sim::components::Model(),
		gz::sim::components::Name(_platformModelName),
		gz::sim::components::ParentEntity(_worldEntity));

	if (droneEntity == gz::sim::kNullEntity || platformEntity == gz::sim::kNullEntity) {
		return false;
	}

	gz::sim::Model droneModel(droneEntity);
	gz::sim::Model platformModel(platformEntity);
	const gz::sim::Entity baseEntity = droneModel.LinkByName(ecm, _droneBaseLinkName);
	const gz::sim::Entity cameraEntity = droneModel.LinkByName(ecm, _cameraLinkName);
	const gz::sim::Entity platformLinkEntity = platformModel.LinkByName(ecm, _platformLinkName);

	if (baseEntity == gz::sim::kNullEntity || cameraEntity == gz::sim::kNullEntity ||
	    platformLinkEntity == gz::sim::kNullEntity) {
		return false;
	}

	_droneModel = droneModel;
	_platformModel = platformModel;
	_droneBaseLink = gz::sim::Link(baseEntity);
	_cameraLink = gz::sim::Link(cameraEntity);
	_platformLink = gz::sim::Link(platformLinkEntity);

	// Gazebo does not guarantee these components are populated until a system asks
	// for them. Enable them once, before reading in PostUpdate.
	for (const gz::sim::Link *link : {&_droneBaseLink, &_cameraLink, &_platformLink}) {
		link->EnableVelocityChecks(ecm, true);
		link->EnableAccelerationChecks(ecm, true);
	}

	_entitiesReady = true;
	gzmsg << "[BeeLandingTruth] resolved drone [" << _droneModelName
	      << "] and platform [" << _platformModelName << "] entities.\n";
	return true;
}

void BeeLandingTruth::clearEntityHandles()
{
	_entitiesReady = false;
	_droneModel = gz::sim::Model(gz::sim::kNullEntity);
	_platformModel = gz::sim::Model(gz::sim::kNullEntity);
	_droneBaseLink = gz::sim::Link(gz::sim::kNullEntity);
	_cameraLink = gz::sim::Link(gz::sim::kNullEntity);
	_platformLink = gz::sim::Link(gz::sim::kNullEntity);
}

void BeeLandingTruth::resetRuntimeState()
{
	_sequence = 0;
	_lastPublishSimTimeSec = nan();
	_previousMinPadDistanceM = nan();
	_previousAnyContact = false;
	_previousLeftContact = false;
	_previousRightContact = false;
	_geometricCrossingLatched = false;
	_contactConfirmed = false;
	_contactStartSimTimeSec = nan();
	_firstGeometricCrossingSimTimeSec = nan();
	_firstAnyContactSimTimeSec = nan();
	_contactConfirmedSimTimeSec = nan();
	{
		std::lock_guard<std::mutex> lock(_contactMutex);
		_leftContact = ContactSnapshot{};
		_rightContact = ContactSnapshot{};
	}
}

std::string BeeLandingTruth::readSdfString(
		const std::shared_ptr<const sdf::Element> &sdf,
		const char *tag,
		const std::string &defaultValue) const
{
	return sdf->HasElement(tag) ? sdf->Get<std::string>(tag) : defaultValue;
}

double BeeLandingTruth::readSdfDouble(
		const std::shared_ptr<const sdf::Element> &sdf,
		const char *tag,
		double defaultValue) const
{
	return sdf->HasElement(tag) ? sdf->Get<double>(tag) : defaultValue;
}

gz::math::Vector3d BeeLandingTruth::readSdfVector3(
		const std::shared_ptr<const sdf::Element> &sdf,
		const char *tag,
		const gz::math::Vector3d &defaultValue) const
{
	return sdf->HasElement(tag) ? sdf->Get<gz::math::Vector3d>(tag) : defaultValue;
}

void BeeLandingTruth::onLeftContacts(const gz::msgs::Contacts &msg)
{
	std::lock_guard<std::mutex> lock(_contactMutex);
	_leftContact = parseContacts(msg);
}

void BeeLandingTruth::onRightContacts(const gz::msgs::Contacts &msg)
{
	std::lock_guard<std::mutex> lock(_contactMutex);
	_rightContact = parseContacts(msg);
}

BeeLandingTruth::ContactSnapshot BeeLandingTruth::parseContacts(
		const gz::msgs::Contacts &msg) const
{
	ContactSnapshot result;
	result.sensorSimTimeSec = messageStampSec(msg);

	for (const auto &contact : msg.contact()) {
		const std::string collision1 = contact.collision1().name();
		const std::string collision2 = contact.collision2().name();
		const bool collision1IsTarget = contains(collision1, _contactTargetSubstring);
		const bool collision2IsTarget = contains(collision2, _contactTargetSubstring);
		if (!collision1IsTarget && !collision2IsTarget) {
			continue;
		}

		result.active = true;
		for (const auto &wrench : contact.wrench()) {
			// The force on the non-platform body is the force applied to the gear.
			const auto &force = collision1IsTarget
				? wrench.body_2_wrench().force()
				: wrench.body_1_wrench().force();
			result.forceMagnitudeN += std::sqrt(
				force.x() * force.x() + force.y() * force.y() + force.z() * force.z());
		}
	}
	return result;
}

double BeeLandingTruth::messageStampSec(const gz::msgs::Contacts &msg)
{
	if (!msg.has_header() || !msg.header().has_stamp()) {
		return nan();
	}
	return timeMessageSec(msg.header().stamp());
}

double BeeLandingTruth::timeMessageSec(const gz::msgs::Time &time)
{
	return static_cast<double>(time.sec()) + 1e-9 * static_cast<double>(time.nsec());
}

double BeeLandingTruth::nan()
{
	return std::numeric_limits<double>::quiet_NaN();
}

bool BeeLandingTruth::finite(double value)
{
	return std::isfinite(value);
}

std::optional<gz::math::Vector3d> BeeLandingTruth::platformPointVelocity(
		const gz::math::Vector3d &pointWorld,
		const gz::math::Pose3d &platformPose,
		const gz::sim::EntityComponentManager &ecm) const
{
	const gz::math::Vector3d offsetWorld = pointWorld - platformPose.Pos();
	const gz::math::Vector3d offsetBody = platformPose.Rot().RotateVectorReverse(offsetWorld);
	return _platformLink.WorldLinearVelocity(ecm, offsetBody);
}

BeeLandingTruth::PointTruth BeeLandingTruth::minimumSkidTruth(
		const gz::math::Vector3d &skidCenterBase,
		const gz::math::Pose3d &dronePose,
		const gz::math::Pose3d &platformPose,
		const gz::math::Vector3d &deckPointWorld,
		const gz::math::Vector3d &deckNormalWorld,
		const gz::sim::EntityComponentManager &ecm) const
{
	PointTruth best;
	bool haveBest = false;
	const std::array<double, 2> xOffsets{-_skidHalfLengthM, _skidHalfLengthM};
	const std::array<double, 2> yOffsets{-_skidHalfWidthM, _skidHalfWidthM};

	for (const double dx : xOffsets) {
		for (const double dy : yOffsets) {
			const gz::math::Vector3d pointBase =
				skidCenterBase + gz::math::Vector3d(dx, dy, 0.0);
			const gz::math::Vector3d pointWorld =
				dronePose.Pos() + dronePose.Rot().RotateVector(pointBase);
			const auto velocity = _droneBaseLink.WorldLinearVelocity(ecm, pointBase);
			if (!velocity.has_value()) {
				continue;
			}
			const PointTruth candidate = pointRelativeToDeck(
				pointWorld, *velocity, platformPose, deckPointWorld, deckNormalWorld, ecm);
			if (!haveBest || candidate.signedDistanceM < best.signedDistanceM) {
				best = candidate;
				haveBest = true;
			}
		}
	}
	return best;
}

BeeLandingTruth::PointTruth BeeLandingTruth::pointRelativeToDeck(
		const gz::math::Vector3d &pointWorld,
		const gz::math::Vector3d &pointVelocityWorld,
		const gz::math::Pose3d &platformPose,
		const gz::math::Vector3d &deckPointWorld,
		const gz::math::Vector3d &deckNormalWorld,
		const gz::sim::EntityComponentManager &ecm) const
{
	PointTruth result;
	result.position = pointWorld;
	result.velocity = pointVelocityWorld;
	result.signedDistanceM = deckNormalWorld.Dot(pointWorld - deckPointWorld);

	const gz::math::Vector3d projectedDeckPoint =
		pointWorld - result.signedDistanceM * deckNormalWorld;
	const auto deckVelocity = platformPointVelocity(projectedDeckPoint, platformPose, ecm);
	if (deckVelocity.has_value()) {
		// Positive means the point is moving toward the deck.
		result.closingRateMS = -deckNormalWorld.Dot(pointVelocityWorld - *deckVelocity);
	}
	return result;
}

void BeeLandingTruth::publishTruth(const gz::sim::UpdateInfo &info,
		const gz::sim::EntityComponentManager &ecm,
		bool forcePublish)
{
	const double simTimeSec = std::chrono::duration<double>(info.simTime).count();
	const double physicsDtSec = std::chrono::duration<double>(info.dt).count();

	const auto dronePoseOpt = _droneBaseLink.WorldPose(ecm);
	const auto cameraPoseOpt = _cameraLink.WorldPose(ecm);
	const auto platformPoseOpt = _platformLink.WorldPose(ecm);
	const auto droneVelocityOpt = _droneBaseLink.WorldLinearVelocity(ecm);
	const auto droneAngularVelocityOpt = _droneBaseLink.WorldAngularVelocity(ecm);
	const auto droneAccelerationOpt = _droneBaseLink.WorldLinearAcceleration(ecm);
	const auto cameraVelocityOpt = _cameraLink.WorldLinearVelocity(ecm);
	const auto platformVelocityOpt = _platformLink.WorldLinearVelocity(ecm);
	const auto platformAngularVelocityOpt = _platformLink.WorldAngularVelocity(ecm);
	const auto platformAccelerationOpt = _platformLink.WorldLinearAcceleration(ecm);

	if (!dronePoseOpt.has_value() || !cameraPoseOpt.has_value() || !platformPoseOpt.has_value() ||
	    !droneVelocityOpt.has_value() || !cameraVelocityOpt.has_value() ||
	    !platformVelocityOpt.has_value()) {
		return;
	}

	const gz::math::Pose3d &dronePose = *dronePoseOpt;
	const gz::math::Pose3d &cameraPose = *cameraPoseOpt;
	const gz::math::Pose3d &platformPose = *platformPoseOpt;

	const gz::math::Vector3d deckOffsetBody(0.0, 0.0, _platformTopOffsetM);
	const gz::math::Vector3d deckPointWorld =
		platformPose.Pos() + platformPose.Rot().RotateVector(deckOffsetBody);
	const gz::math::Vector3d deckNormalWorld =
		platformPose.Rot().RotateVector(gz::math::Vector3d::UnitZ).Normalized();

	// Each skid is a 0.25 x 0.015 m rail. The first geometric touch under
	// roll / pitch occurs at a bottom corner, not necessarily at the rail centre,
	// so evaluate all four bottom corners and retain the lowest one per skid.
	const PointTruth leftPad = minimumSkidTruth(
		_leftPadPointBase, dronePose, platformPose, deckPointWorld, deckNormalWorld, ecm);
	const PointTruth rightPad = minimumSkidTruth(
		_rightPadPointBase, dronePose, platformPose, deckPointWorld, deckNormalWorld, ecm);
	if (!finite(leftPad.signedDistanceM) || !finite(rightPad.signedDistanceM)) {
		return;
	}
	const PointTruth camera = pointRelativeToDeck(
		cameraPose.Pos(), *cameraVelocityOpt, platformPose, deckPointWorld, deckNormalWorld, ecm);

	const bool leftIsMinimum = leftPad.signedDistanceM <= rightPad.signedDistanceM;
	const double minPadDistance = leftIsMinimum
		? leftPad.signedDistanceM : rightPad.signedDistanceM;
	const double contactPadClosingRate = leftIsMinimum
		? leftPad.closingRateMS : rightPad.closingRateMS;

	const gz::math::Vector3d cameraForwardWorld =
		cameraPose.Rot().RotateVector(gz::math::Vector3d::UnitX).Normalized();
	const double viewAlignment = -cameraForwardWorld.Dot(deckNormalWorld);
	const double opticalRange = viewAlignment > _minimumViewAlignment
		? camera.signedDistanceM / viewAlignment : nan();
	const bool divergenceValid =
		camera.signedDistanceM > _divergenceMinDistanceM &&
		viewAlignment > _minimumViewAlignment &&
		finite(camera.closingRateMS);
	const double normalExpansionRate = divergenceValid
		? camera.closingRateMS / camera.signedDistanceM : nan();
	// For a fronto-parallel plane under translation, the mathematical 2-D
	// image-flow divergence du/dx + dv/dy is twice the one-dimensional
	// expansion / time-to-contact rate c/h. Publish both conventions so the
	// analyser can compare like with like instead of hiding a factor-of-two
	// convention inside the word "divergence".
	const double frontoparallelFlowDivergence = divergenceValid
		? 2.0 * normalExpansionRate : nan();

	ContactSnapshot leftContact;
	ContactSnapshot rightContact;
	{
		std::lock_guard<std::mutex> lock(_contactMutex);
		leftContact = _leftContact;
		rightContact = _rightContact;
	}

	const auto age = [simTimeSec](const ContactSnapshot &snapshot) {
		return finite(snapshot.sensorSimTimeSec)
			? std::max(0.0, simTimeSec - snapshot.sensorSimTimeSec)
			: nan();
	};
	const double leftAgeSec = age(leftContact);
	const double rightAgeSec = age(rightContact);
	const bool leftContactActive = leftContact.active && finite(leftAgeSec) &&
		leftAgeSec <= _contactStaleTimeoutSec;
	const bool rightContactActive = rightContact.active && finite(rightAgeSec) &&
		rightAgeSec <= _contactStaleTimeoutSec;
	const bool anyContact = leftContactActive || rightContactActive;

	bool eventChanged = false;
	if (!_geometricCrossingLatched && finite(minPadDistance) && minPadDistance <= 0.0 &&
	    (!finite(_previousMinPadDistanceM) || _previousMinPadDistanceM > 0.0)) {
		_geometricCrossingLatched = true;
		_firstGeometricCrossingSimTimeSec = simTimeSec;
		eventChanged = true;
	}
	_previousMinPadDistanceM = minPadDistance;

	if (anyContact) {
		if (!finite(_contactStartSimTimeSec)) {
			const double sensorEventTime = std::min(
				leftContactActive && finite(leftContact.sensorSimTimeSec)
					? leftContact.sensorSimTimeSec : simTimeSec,
				rightContactActive && finite(rightContact.sensorSimTimeSec)
					? rightContact.sensorSimTimeSec : simTimeSec);
			_contactStartSimTimeSec = sensorEventTime;
			if (!finite(_firstAnyContactSimTimeSec)) {
				_firstAnyContactSimTimeSec = sensorEventTime;
			}
			eventChanged = true;
		}
		if (!_contactConfirmed &&
		    simTimeSec - _contactStartSimTimeSec >= _contactConfirmationSec) {
			_contactConfirmed = true;
			_contactConfirmedSimTimeSec = simTimeSec;
			eventChanged = true;
		}
	} else if (!_contactConfirmed) {
		_contactStartSimTimeSec = nan();
	}

	if (leftContactActive != _previousLeftContact ||
	    rightContactActive != _previousRightContact ||
	    anyContact != _previousAnyContact) {
		eventChanged = true;
	}
	_previousLeftContact = leftContactActive;
	_previousRightContact = rightContactActive;
	_previousAnyContact = anyContact;

	const double publishPeriodSec = _publishRateHz > 0.0 ? 1.0 / _publishRateHz : 0.0;
	const bool rateDue = !finite(_lastPublishSimTimeSec) || publishPeriodSec <= 0.0 ||
		simTimeSec - _lastPublishSimTimeSec + kTimeEpsilon >= publishPeriodSec;
	if (!forcePublish && !eventChanged && !rateDue) {
		return;
	}

	gz::msgs::Float_V msg;
	for (std::size_t i = 0; i < static_cast<std::size_t>(TruthField::COUNT); ++i) {
		msg.add_data(std::numeric_limits<float>::quiet_NaN());
	}

	set(msg, TruthField::SCHEMA_VERSION, kTruthSchemaVersion);
	set(msg, TruthField::SEQUENCE, static_cast<double>(_sequence));
	set(msg, TruthField::SIM_TIME_SEC, simTimeSec);
	set(msg, TruthField::PHYSICS_DT_SEC, physicsDtSec);
	set(msg, TruthField::ENTITIES_READY, 1.0);

	setPose(msg, TruthField::DRONE_POSITION_X, TruthField::DRONE_ORIENTATION_X, dronePose);
	setVector(msg, TruthField::DRONE_LINEAR_VELOCITY_X, *droneVelocityOpt);
	setVector(msg, TruthField::DRONE_ANGULAR_VELOCITY_X,
		droneAngularVelocityOpt.value_or(gz::math::Vector3d::Zero));
	setVector(msg, TruthField::DRONE_LINEAR_ACCELERATION_X,
		droneAccelerationOpt.value_or(gz::math::Vector3d::Zero));

	setPose(msg, TruthField::PLATFORM_POSITION_X, TruthField::PLATFORM_ORIENTATION_X, platformPose);
	setVector(msg, TruthField::PLATFORM_LINEAR_VELOCITY_X, *platformVelocityOpt);
	setVector(msg, TruthField::PLATFORM_ANGULAR_VELOCITY_X,
		platformAngularVelocityOpt.value_or(gz::math::Vector3d::Zero));
	setVector(msg, TruthField::PLATFORM_LINEAR_ACCELERATION_X,
		platformAccelerationOpt.value_or(gz::math::Vector3d::Zero));

	setVector(msg, TruthField::DECK_POINT_X, deckPointWorld);
	setVector(msg, TruthField::DECK_NORMAL_X, deckNormalWorld);

	setVector(msg, TruthField::LEFT_PAD_POSITION_X, leftPad.position);
	setVector(msg, TruthField::LEFT_PAD_VELOCITY_X, leftPad.velocity);
	set(msg, TruthField::LEFT_PAD_SIGNED_DISTANCE_M, leftPad.signedDistanceM);
	set(msg, TruthField::LEFT_PAD_CLOSING_RATE_M_S, leftPad.closingRateMS);

	setVector(msg, TruthField::RIGHT_PAD_POSITION_X, rightPad.position);
	setVector(msg, TruthField::RIGHT_PAD_VELOCITY_X, rightPad.velocity);
	set(msg, TruthField::RIGHT_PAD_SIGNED_DISTANCE_M, rightPad.signedDistanceM);
	set(msg, TruthField::RIGHT_PAD_CLOSING_RATE_M_S, rightPad.closingRateMS);
	set(msg, TruthField::MIN_PAD_SIGNED_DISTANCE_M, minPadDistance);
	set(msg, TruthField::CONTACT_PAD_CLOSING_RATE_M_S, contactPadClosingRate);

	setVector(msg, TruthField::CAMERA_POSITION_X, camera.position);
	setVector(msg, TruthField::CAMERA_VELOCITY_X, camera.velocity);
	set(msg, TruthField::CAMERA_NORMAL_DISTANCE_M, camera.signedDistanceM);
	set(msg, TruthField::CAMERA_NORMAL_CLOSING_RATE_M_S, camera.closingRateMS);
	set(msg, TruthField::CAMERA_OPTICAL_RANGE_M, opticalRange);
	set(msg, TruthField::CAMERA_VIEW_ALIGNMENT, viewAlignment);
	set(msg, TruthField::NORMAL_EXPANSION_RATE_1_S, normalExpansionRate);
	set(msg, TruthField::FRONTPARALLEL_FLOW_DIVERGENCE_1_S,
		frontoparallelFlowDivergence);
	set(msg, TruthField::EXPANSION_TRUTH_VALID, divergenceValid ? 1.0 : 0.0);

	set(msg, TruthField::LEFT_CONTACT, leftContactActive ? 1.0 : 0.0);
	set(msg, TruthField::RIGHT_CONTACT, rightContactActive ? 1.0 : 0.0);
	set(msg, TruthField::ANY_CONTACT, anyContact ? 1.0 : 0.0);
	set(msg, TruthField::CONTACT_CONFIRMED, _contactConfirmed ? 1.0 : 0.0);
	set(msg, TruthField::LEFT_CONTACT_FORCE_N, leftContact.forceMagnitudeN);
	set(msg, TruthField::RIGHT_CONTACT_FORCE_N, rightContact.forceMagnitudeN);
	set(msg, TruthField::LEFT_CONTACT_SENSOR_SIM_TIME_SEC, leftContact.sensorSimTimeSec);
	set(msg, TruthField::RIGHT_CONTACT_SENSOR_SIM_TIME_SEC, rightContact.sensorSimTimeSec);
	set(msg, TruthField::LEFT_CONTACT_AGE_SEC, leftAgeSec);
	set(msg, TruthField::RIGHT_CONTACT_AGE_SEC, rightAgeSec);
	set(msg, TruthField::CONTACT_DWELL_SEC,
		anyContact && finite(_contactStartSimTimeSec)
			? std::max(0.0, simTimeSec - _contactStartSimTimeSec) : 0.0);

	set(msg, TruthField::GEOMETRIC_CROSSING_LATCHED,
		_geometricCrossingLatched ? 1.0 : 0.0);
	set(msg, TruthField::FIRST_GEOMETRIC_CROSSING_SIM_TIME_SEC,
		_firstGeometricCrossingSimTimeSec);
	set(msg, TruthField::FIRST_ANY_CONTACT_SIM_TIME_SEC, _firstAnyContactSimTimeSec);
	set(msg, TruthField::CONTACT_CONFIRMED_SIM_TIME_SEC, _contactConfirmedSimTimeSec);

	_truthPublisher.Publish(msg);
	++_sequence;
	_lastPublishSimTimeSec = simTimeSec;
}

void BeeLandingTruth::set(gz::msgs::Float_V &msg, TruthField field, double value)
{
	msg.set_data(static_cast<int>(field), asFloat(value));
}

void BeeLandingTruth::setVector(gz::msgs::Float_V &msg, TruthField firstField,
		const gz::math::Vector3d &value)
{
	const std::size_t first = static_cast<std::size_t>(firstField);
	msg.set_data(static_cast<int>(first + 0), asFloat(value.X()));
	msg.set_data(static_cast<int>(first + 1), asFloat(value.Y()));
	msg.set_data(static_cast<int>(first + 2), asFloat(value.Z()));
}

void BeeLandingTruth::setPose(gz::msgs::Float_V &msg,
		TruthField positionX,
		TruthField orientationX,
		const gz::math::Pose3d &pose)
{
	setVector(msg, positionX, pose.Pos());
	const std::size_t first = static_cast<std::size_t>(orientationX);
	msg.set_data(static_cast<int>(first + 0), asFloat(pose.Rot().X()));
	msg.set_data(static_cast<int>(first + 1), asFloat(pose.Rot().Y()));
	msg.set_data(static_cast<int>(first + 2), asFloat(pose.Rot().Z()));
	msg.set_data(static_cast<int>(first + 3), asFloat(pose.Rot().W()));
}
