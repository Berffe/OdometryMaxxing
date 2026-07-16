/****************************************************************************
 *
 *   Copyright (c) 2025 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#include "MovingPlatformController.hpp"

#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>

#include <gz/sim/components/Inertial.hh>

using namespace custom;

GZ_ADD_PLUGIN(
	OscillatingPlatformController,
	gz::sim::System,
	gz::sim::ISystemConfigure,
	gz::sim::ISystemPreUpdate,
	gz::sim::ISystemPostUpdate
)

void OscillatingPlatformController::Configure(const gz::sim::Entity &entity,
		const std::shared_ptr<const sdf::Element> &sdf,
		gz::sim::EntityComponentManager &ecm,
		gz::sim::EventManager &eventMgr)
{
	(void)eventMgr;

	_model = gz::sim::Model(entity);

	const std::string link_name = readSdfString(sdf, "link_name", "platform_link");
	const gz::sim::Entity link_entity = _model.LinkByName(ecm, link_name);

	if (link_entity == gz::sim::kNullEntity) {
		throw std::runtime_error("OscillatingPlatformController::Configure: link \"" + link_name + "\" was not found.");
	}

	_link = gz::sim::Link(link_entity);

	// Needed for WorldPose / velocity queries to be available reliably from the
	// link API. The platform is now driven by link velocity, so the physics solver
	// can resolve contacts against the moving deck collision.
	_link.EnableVelocityChecks(ecm, true);

	const auto link_pose = _link.WorldPose(ecm);
	_initial_pose = link_pose.has_value() ? *link_pose : gz::sim::worldPose(entity, ecm);

	_x_amplitude = readSdfDouble(sdf, "x_amplitude", _x_amplitude);
	_x_frequency = readSdfDouble(sdf, "x_frequency", _x_frequency);
	_x_phase = readSdfDouble(sdf, "x_phase", _x_phase);

	_y_amplitude = readSdfDouble(sdf, "y_amplitude", _y_amplitude);
	_y_frequency = readSdfDouble(sdf, "y_frequency", _y_frequency);
	_y_phase = readSdfDouble(sdf, "y_phase", _y_phase);

	_z_amplitude = readSdfDouble(sdf, "z_amplitude", _z_amplitude);
	_z_frequency = readSdfDouble(sdf, "z_frequency", _z_frequency);
	_z_phase = readSdfDouble(sdf, "z_phase", _z_phase);

	_position_gain = readSdfDouble(sdf, "position_gain", _position_gain);
	_max_linear_velocity = readSdfDouble(sdf, "max_linear_velocity", _max_linear_velocity);
	_velocity_tracking_gain = readSdfDouble(sdf, "velocity_tracking_gain", _velocity_tracking_gain);
	_fallback_mass_kg = readSdfDouble(sdf, "fallback_mass_kg", _fallback_mass_kg);
	_gravity_magnitude = readSdfDouble(sdf, "gravity_magnitude", _gravity_magnitude);

	// Optional <pose_topic> SDF tag; defaults to /platform/pose.
	_pose_topic = readSdfString(sdf, "pose_topic", _pose_topic);
	_pose_publisher = _transport_node.Advertise<gz::msgs::Pose>(_pose_topic);
}

void OscillatingPlatformController::PreUpdate(const gz::sim::UpdateInfo &info,
		gz::sim::EntityComponentManager &ecm)
{
	if (info.paused) {
		return;
	}

	if (_link.Entity() == gz::sim::kNullEntity) {
		return;
	}

	const double t = std::chrono::duration<double>(info.simTime).count();

	const gz::math::Vector3d target_position = sinusoidalPosition(t);
	const gz::math::Vector3d target_velocity = sinusoidalVelocity(t);

	const auto current_pose = _link.WorldPose(ecm);
	if (!current_pose.has_value()) {
		return;
	}

	const gz::math::Vector3d position_error = target_position - current_pose->Pos();
	gz::math::Vector3d commanded_velocity = target_velocity + _position_gain * position_error;
	commanded_velocity = clampVectorNorm(commanded_velocity, _max_linear_velocity);

	// ONE-WAY COUPLING: the deck drives the drone, the drone (essentially) does
	// not drive the deck -- but the deck must still be a DYNAMIC body so contact
	// generates a real normal force on the landing gear.
	//
	// The previous SetLinearVelocity() made the link KINEMATIC: its velocity was
	// overwritten every tick, so the solver's contact correction (which would push
	// the drone back out) was deleted before it could act. The deck was an
	// infinitely-stiff treadmill -- it passed straight through the legs, so the
	// drone felt nothing and sank ~0.2 m into the deck before the motor-cut (fired
	// only after TouchPlugin) stopped it. That ~0.2 m at the descent rate is the
	// bulk of the touchdown-to-detection gap.
	//
	// Now the servo is applied as a velocity-tracking FORCE instead. The link is a
	// normal dynamic body, so contact impulses are respected and the legs stop at
	// the surface. The platform is heavy (mass 1000 kg) and the servo is stiff, so
	// the drone's reaction on the deck is negligible and transient: a ~25 N leg
	// load perturbs a 1000 kg deck by ~F/(M*position_gain) ~ 3 mm of position,
	// which the servo restores within ~1/position_gain seconds. The deck's
	// commanded trajectory is therefore unaffected in practice -- the coupling is
	// one-way exactly as intended, but through real contact mechanics rather than
	// by teleporting through the drone.
	const auto current_velocity = _link.WorldLinearVelocity(ecm);
	const gz::math::Vector3d actual_velocity =
		current_velocity.has_value() ? *current_velocity : gz::math::Vector3d::Zero;

	const std::optional<double> mass = linkMass(ecm);
	const double m = (mass.has_value() && *mass > 0.0) ? *mass : _fallback_mass_kg;

	// Critically-damped-ish velocity tracking: F = m * gain_v * (v_cmd - v_actual).
	// gain_v sets how hard the deck insists on its commanded velocity; high enough
	// to hold the sinusoid tightly, finite so contact can momentarily deflect it by
	// the few-mm quantified above instead of being ignored outright.
	const gz::math::Vector3d velocity_error = commanded_velocity - actual_velocity;
	const gz::math::Vector3d tracking_force = m * _velocity_tracking_gain * velocity_error;

	// Cancel gravity so the servo does not have to fight it -- IF gravity were
	// enabled on this link. It is not: bee_platform.sdf's platform_link carries
	// <gravity>false</gravity> (set back when the deck was purely kinematic via
	// SetLinearVelocity, and never revisited when this method switched to a real
	// force). Gazebo therefore never applies gravity to this link in the first
	// place, so adding a compensating m*g force here was cancelling something
	// that was never there: a genuine, CONSTANT, unbalanced ~m*g upward force
	// (9800 N at mass=1000) that the velocity servo then had to fight every
	// tick. With commanded_velocity clamped at _max_linear_velocity, a
	// sustained disturbance that size is exactly the kind of thing that can
	// push a clamped feedback loop into a large, irregular limit cycle rather
	// than a clean tracking error -- consistent with what was observed on a
	// real log: platform_z_m's actual amplitude/period (~0.8 m, an irregular
	// ~12-24 s spacing between peaks) bore no resemblance to the commanded
	// sinusoid (0.3 m at 0.3 Hz, i.e. a 3.33 s period).
	//
	// Fix: no gravity_comp term. If gravity is ever re-enabled on this link
	// (<gravity>true</gravity>), this compensation must come back -- the two
	// must always be changed together, not independently.
	_link.AddWorldForce(ecm, tracking_force);
	_link.SetAngularVelocity(ecm, gz::math::Vector3d::Zero);
}

void OscillatingPlatformController::PostUpdate(const gz::sim::UpdateInfo &info,
		const gz::sim::EntityComponentManager &ecm)
{
	if (info.paused) {
		return;
	}

	if (_link.Entity() == gz::sim::kNullEntity) {
		return;
	}

	const auto pose = _link.WorldPose(ecm);
	if (pose.has_value()) {
		publishPose(*pose);
	}
}

double OscillatingPlatformController::readSdfDouble(const std::shared_ptr<const sdf::Element> &sdf,
		const char *tag,
		double default_value) const
{
	if (!sdf->HasElement(tag)) {
		return default_value;
	}

	return sdf->Get<double>(tag);
}

std::string OscillatingPlatformController::readSdfString(const std::shared_ptr<const sdf::Element> &sdf,
		const char *tag,
		const std::string &default_value) const
{
	if (!sdf->HasElement(tag)) {
		return default_value;
	}

	return sdf->Get<std::string>(tag);
}

gz::math::Vector3d OscillatingPlatformController::sinusoidalPosition(double time_sec) const
{
	const double x = _initial_pose.Pos().X()
			+ _x_amplitude * std::sin(2.0 * GZ_PI * _x_frequency * time_sec + _x_phase);
	const double y = _initial_pose.Pos().Y()
			+ _y_amplitude * std::sin(2.0 * GZ_PI * _y_frequency * time_sec + _y_phase);
	const double z = _initial_pose.Pos().Z()
			+ _z_amplitude * std::sin(2.0 * GZ_PI * _z_frequency * time_sec + _z_phase);

	return {x, y, z};
}

gz::math::Vector3d OscillatingPlatformController::sinusoidalVelocity(double time_sec) const
{
	const double vx = _x_amplitude * 2.0 * GZ_PI * _x_frequency
			* std::cos(2.0 * GZ_PI * _x_frequency * time_sec + _x_phase);
	const double vy = _y_amplitude * 2.0 * GZ_PI * _y_frequency
			* std::cos(2.0 * GZ_PI * _y_frequency * time_sec + _y_phase);
	const double vz = _z_amplitude * 2.0 * GZ_PI * _z_frequency
			* std::cos(2.0 * GZ_PI * _z_frequency * time_sec + _z_phase);

	return {vx, vy, vz};
}

std::optional<double> OscillatingPlatformController::linkMass(const gz::sim::EntityComponentManager &ecm) const
{
	// The velocity-tracking force needs the deck's mass (F = m*gain*dv). Read it
	// from the inertial component so the plugin stays correct if the SDF <mass>
	// changes; falls back to _fallback_mass_kg if unavailable. Cached after the
	// first successful read -- mass is constant.
	if (_cached_mass.has_value()) {
		return _cached_mass;
	}
	const auto inertial = ecm.Component<gz::sim::components::Inertial>(_link.Entity());
	if (inertial != nullptr) {
		_cached_mass = inertial->Data().MassMatrix().Mass();
	}
	return _cached_mass;
}

gz::math::Vector3d OscillatingPlatformController::clampVectorNorm(const gz::math::Vector3d &value, double max_norm) const
{
	if (max_norm <= 0.0) {
		return value;
	}

	const double norm = value.Length();
	if (norm <= max_norm || norm <= 1e-12) {
		return value;
	}

	return value * (max_norm / norm);
}

void OscillatingPlatformController::publishPose(const gz::math::Pose3d &pose)
{
	gz::msgs::Pose pose_msg;
	pose_msg.set_name("platform");

	pose_msg.mutable_position()->set_x(pose.Pos().X());
	pose_msg.mutable_position()->set_y(pose.Pos().Y());
	pose_msg.mutable_position()->set_z(pose.Pos().Z());

	pose_msg.mutable_orientation()->set_x(pose.Rot().X());
	pose_msg.mutable_orientation()->set_y(pose.Rot().Y());
	pose_msg.mutable_orientation()->set_z(pose.Rot().Z());
	pose_msg.mutable_orientation()->set_w(pose.Rot().W());

	_pose_publisher.Publish(pose_msg);
}
