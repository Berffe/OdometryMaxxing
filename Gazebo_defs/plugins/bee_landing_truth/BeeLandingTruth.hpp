/****************************************************************************
 *
 *   BEE_LAND Gazebo-native truth system
 *
 ****************************************************************************/

#pragma once

#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/contacts.pb.h>
#include <gz/msgs/float_v.pb.h>
#include <gz/msgs/time.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/World.hh>
#include <gz/transport/Node.hh>

#include <sdf/Element.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace custom
{

/// Fixed layout of /bee_land/truth (gz.msgs.Float_V).
///
/// All calculations are performed in double precision inside Gazebo. Values
/// are cast to float only for this bridge-friendly transport envelope. The
/// layout is versioned by SCHEMA_VERSION and mirrored in truth_layout.py.
enum class TruthField : std::size_t
{
	SCHEMA_VERSION = 0,
	SEQUENCE,
	SIM_TIME_SEC,
	PHYSICS_DT_SEC,
	ENTITIES_READY,

	DRONE_POSITION_X,
	DRONE_POSITION_Y,
	DRONE_POSITION_Z,
	DRONE_ORIENTATION_X,
	DRONE_ORIENTATION_Y,
	DRONE_ORIENTATION_Z,
	DRONE_ORIENTATION_W,
	DRONE_LINEAR_VELOCITY_X,
	DRONE_LINEAR_VELOCITY_Y,
	DRONE_LINEAR_VELOCITY_Z,
	DRONE_ANGULAR_VELOCITY_X,
	DRONE_ANGULAR_VELOCITY_Y,
	DRONE_ANGULAR_VELOCITY_Z,
	DRONE_LINEAR_ACCELERATION_X,
	DRONE_LINEAR_ACCELERATION_Y,
	DRONE_LINEAR_ACCELERATION_Z,

	PLATFORM_POSITION_X,
	PLATFORM_POSITION_Y,
	PLATFORM_POSITION_Z,
	PLATFORM_ORIENTATION_X,
	PLATFORM_ORIENTATION_Y,
	PLATFORM_ORIENTATION_Z,
	PLATFORM_ORIENTATION_W,
	PLATFORM_LINEAR_VELOCITY_X,
	PLATFORM_LINEAR_VELOCITY_Y,
	PLATFORM_LINEAR_VELOCITY_Z,
	PLATFORM_ANGULAR_VELOCITY_X,
	PLATFORM_ANGULAR_VELOCITY_Y,
	PLATFORM_ANGULAR_VELOCITY_Z,
	PLATFORM_LINEAR_ACCELERATION_X,
	PLATFORM_LINEAR_ACCELERATION_Y,
	PLATFORM_LINEAR_ACCELERATION_Z,

	DECK_POINT_X,
	DECK_POINT_Y,
	DECK_POINT_Z,
	DECK_NORMAL_X,
	DECK_NORMAL_Y,
	DECK_NORMAL_Z,

	LEFT_PAD_POSITION_X,
	LEFT_PAD_POSITION_Y,
	LEFT_PAD_POSITION_Z,
	LEFT_PAD_VELOCITY_X,
	LEFT_PAD_VELOCITY_Y,
	LEFT_PAD_VELOCITY_Z,
	LEFT_PAD_SIGNED_DISTANCE_M,
	LEFT_PAD_CLOSING_RATE_M_S,

	RIGHT_PAD_POSITION_X,
	RIGHT_PAD_POSITION_Y,
	RIGHT_PAD_POSITION_Z,
	RIGHT_PAD_VELOCITY_X,
	RIGHT_PAD_VELOCITY_Y,
	RIGHT_PAD_VELOCITY_Z,
	RIGHT_PAD_SIGNED_DISTANCE_M,
	RIGHT_PAD_CLOSING_RATE_M_S,

	MIN_PAD_SIGNED_DISTANCE_M,
	CONTACT_PAD_CLOSING_RATE_M_S,

	CAMERA_POSITION_X,
	CAMERA_POSITION_Y,
	CAMERA_POSITION_Z,
	CAMERA_VELOCITY_X,
	CAMERA_VELOCITY_Y,
	CAMERA_VELOCITY_Z,
	CAMERA_NORMAL_DISTANCE_M,
	CAMERA_NORMAL_CLOSING_RATE_M_S,
	CAMERA_OPTICAL_RANGE_M,
	CAMERA_VIEW_ALIGNMENT,
	NORMAL_EXPANSION_RATE_1_S,
	FRONTPARALLEL_FLOW_DIVERGENCE_1_S,
	EXPANSION_TRUTH_VALID,

	LEFT_CONTACT,
	RIGHT_CONTACT,
	ANY_CONTACT,
	CONTACT_CONFIRMED,
	LEFT_CONTACT_FORCE_N,
	RIGHT_CONTACT_FORCE_N,
	LEFT_CONTACT_SENSOR_SIM_TIME_SEC,
	RIGHT_CONTACT_SENSOR_SIM_TIME_SEC,
	LEFT_CONTACT_AGE_SEC,
	RIGHT_CONTACT_AGE_SEC,
	CONTACT_DWELL_SEC,

	GEOMETRIC_CROSSING_LATCHED,
	FIRST_GEOMETRIC_CROSSING_SIM_TIME_SEC,
	FIRST_ANY_CONTACT_SIM_TIME_SEC,
	CONTACT_CONFIRMED_SIM_TIME_SEC,

	COUNT
};

class BeeLandingTruth:
	public gz::sim::System,
	public gz::sim::ISystemConfigure,
	public gz::sim::ISystemPreUpdate,
	public gz::sim::ISystemPostUpdate
{
public:
	void Configure(const gz::sim::Entity &entity,
			const std::shared_ptr<const sdf::Element> &sdf,
			gz::sim::EntityComponentManager &ecm,
			gz::sim::EventManager &eventMgr) override;

	void PreUpdate(const gz::sim::UpdateInfo &info,
			gz::sim::EntityComponentManager &ecm) final;

	void PostUpdate(const gz::sim::UpdateInfo &info,
			const gz::sim::EntityComponentManager &ecm) final;

private:
	struct ContactSnapshot
	{
		bool active{false};
		double forceMagnitudeN{0.0};
		double sensorSimTimeSec{std::numeric_limits<double>::quiet_NaN()};
	};

	struct PointTruth
	{
		gz::math::Vector3d position{gz::math::Vector3d::Zero};
		gz::math::Vector3d velocity{gz::math::Vector3d::Zero};
		double signedDistanceM{std::numeric_limits<double>::quiet_NaN()};
		double closingRateMS{std::numeric_limits<double>::quiet_NaN()};
	};

	bool resolveEntities(gz::sim::EntityComponentManager &ecm);
	void clearEntityHandles();
	void resetRuntimeState();

	std::string readSdfString(const std::shared_ptr<const sdf::Element> &sdf,
			const char *tag,
			const std::string &defaultValue) const;
	double readSdfDouble(const std::shared_ptr<const sdf::Element> &sdf,
			const char *tag,
			double defaultValue) const;
	gz::math::Vector3d readSdfVector3(const std::shared_ptr<const sdf::Element> &sdf,
			const char *tag,
			const gz::math::Vector3d &defaultValue) const;

	void onLeftContacts(const gz::msgs::Contacts &msg);
	void onRightContacts(const gz::msgs::Contacts &msg);
	ContactSnapshot parseContacts(const gz::msgs::Contacts &msg) const;

	static double messageStampSec(const gz::msgs::Contacts &msg);
	static double timeMessageSec(const gz::msgs::Time &time);
	static double nan();
	static bool finite(double value);

	PointTruth minimumSkidTruth(
			const gz::math::Vector3d &skidCenterBase,
			const gz::math::Pose3d &dronePose,
			const gz::math::Pose3d &platformPose,
			const gz::math::Vector3d &deckPointWorld,
			const gz::math::Vector3d &deckNormalWorld,
			const gz::sim::EntityComponentManager &ecm) const;

	PointTruth pointRelativeToDeck(
			const gz::math::Vector3d &pointWorld,
			const gz::math::Vector3d &pointVelocityWorld,
			const gz::math::Pose3d &platformPose,
			const gz::math::Vector3d &deckPointWorld,
			const gz::math::Vector3d &deckNormalWorld,
			const gz::sim::EntityComponentManager &ecm) const;

	std::optional<gz::math::Vector3d> platformPointVelocity(
			const gz::math::Vector3d &pointWorld,
			const gz::math::Pose3d &platformPose,
			const gz::sim::EntityComponentManager &ecm) const;

	void publishTruth(const gz::sim::UpdateInfo &info,
			const gz::sim::EntityComponentManager &ecm,
			bool forcePublish);

	static void set(gz::msgs::Float_V &msg, TruthField field, double value);
	static void setVector(gz::msgs::Float_V &msg, TruthField firstField,
			const gz::math::Vector3d &value);
	static void setPose(gz::msgs::Float_V &msg, TruthField positionX,
			TruthField orientationX, const gz::math::Pose3d &pose);

	gz::sim::Entity _worldEntity{gz::sim::kNullEntity};
	std::string _worldName{"bee_platform"};

	std::string _droneModelName{"bee_x500_0"};
	std::string _platformModelName{"bee_platform"};
	std::string _droneBaseLinkName{"base_link"};
	std::string _cameraLinkName{"bee_camera_link"};
	std::string _platformLinkName{"platform_link"};
	std::string _gearLinkName{"bee_gear_contact_link"};
	std::string _leftSensorName{"landing_gear_contact_sensor_left"};
	std::string _rightSensorName{"landing_gear_contact_sensor_right"};
	std::string _contactTargetSubstring{"bee_platform"};

	std::string _truthTopic{"/bee_land/truth"};
	std::string _leftContactTopic;
	std::string _rightContactTopic;

	gz::math::Vector3d _leftPadPointBase{0.0, -0.132, -0.227};
	gz::math::Vector3d _rightPadPointBase{0.0, 0.132, -0.227};
	double _skidHalfLengthM{0.125};
	double _skidHalfWidthM{0.0075};
	double _platformTopOffsetM{0.10};
	double _publishRateHz{100.0};
	double _contactConfirmationSec{0.04};
	double _contactStaleTimeoutSec{0.03};
	double _divergenceMinDistanceM{0.05};
	double _minimumViewAlignment{0.90};

	gz::sim::Model _droneModel{gz::sim::kNullEntity};
	gz::sim::Model _platformModel{gz::sim::kNullEntity};
	gz::sim::Link _droneBaseLink{gz::sim::kNullEntity};
	gz::sim::Link _cameraLink{gz::sim::kNullEntity};
	gz::sim::Link _platformLink{gz::sim::kNullEntity};
	bool _entitiesReady{false};

	mutable std::mutex _contactMutex;
	ContactSnapshot _leftContact;
	ContactSnapshot _rightContact;

	// Declared after callback state so destruction happens first (reverse member
	// order), preventing a transport callback from outliving its mutex/state.
	gz::transport::Node _transportNode;
	gz::transport::Node::Publisher _truthPublisher;

	uint64_t _sequence{0};
	double _lastObservedSimTimeSec{nan()};
	double _lastPublishSimTimeSec{nan()};
	double _previousMinPadDistanceM{nan()};
	bool _previousAnyContact{false};
	bool _previousLeftContact{false};
	bool _previousRightContact{false};

	bool _geometricCrossingLatched{false};
	bool _contactConfirmed{false};
	double _contactStartSimTimeSec{nan()};
	double _firstGeometricCrossingSimTimeSec{nan()};
	double _firstAnyContactSimTimeSec{nan()};
	double _contactConfirmedSimTimeSec{nan()};
};

} // namespace custom
