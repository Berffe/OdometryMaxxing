/****************************************************************************
 *
 *   Copyright (c) 2025 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#pragma once

#include <gz/math.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/transport/Node.hh>

#include <sdf/Element.hh>

#include <optional>
#include <string>

namespace custom
{
class OscillatingPlatformController:
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
	double readSdfDouble(const std::shared_ptr<const sdf::Element> &sdf,
				const char *tag,
				double default_value) const;

	std::string readSdfString(const std::shared_ptr<const sdf::Element> &sdf,
				const char *tag,
				const std::string &default_value) const;

	gz::math::Vector3d sinusoidalPosition(double time_sec) const;
	gz::math::Vector3d sinusoidalVelocity(double time_sec) const;
	gz::math::Vector3d clampVectorNorm(const gz::math::Vector3d &value, double max_norm) const;

	// Deck mass for the velocity-tracking force (F = m*gain*dv); cached, constant.
	std::optional<double> linkMass(const gz::sim::EntityComponentManager &ecm) const;

	// Publishes the ACTUAL simulated link pose, not only the commanded target.
	// This topic is intentionally single-entity: every message on _pose_topic is
	// the platform pose, so the ROS side can bridge it as gz.msgs.Pose ->
	// geometry_msgs/msg/Pose without relying on generic scene pose broadcasts or
	// entity-name matching.
	void publishPose(const gz::math::Pose3d &pose);

	gz::sim::Model _model{gz::sim::kNullEntity};
	gz::sim::Link _link{gz::sim::kNullEntity};
	gz::math::Pose3d _initial_pose{0., 0., 0., 0., 0., 0.};

	double _x_amplitude{1.0};
	double _x_frequency{0.10};
	double _x_phase{0.0};

	double _y_amplitude{0.0};
	double _y_frequency{0.0};
	double _y_phase{0.0};

	double _z_amplitude{0.30};
	double _z_frequency{0.20};
	double _z_phase{0.0};

	// Velocity-driven tracking parameters. The platform follows the desired
	// sinusoid through physics instead of being teleported with SetWorldPoseCmd.
	// commanded_velocity = feedforward_sinusoid_velocity + position_gain * error.
	double _position_gain{8.0};
	double _max_linear_velocity{5.0};

	// Velocity-tracking-force parameters (see PreUpdate). The deck is now a DYNAMIC
	// body held on its sinusoid by a stiff velocity servo instead of a kinematic
	// SetLinearVelocity, so contact with the landing gear produces a real normal
	// force and the drone stops at the surface instead of sinking through it. The
	// coupling stays one-way in practice: at mass 1000 kg a drone leg load
	// perturbs the deck by only a few millimetres, restored within ~1/position_gain
	// seconds. _velocity_tracking_gain [1/s] sets servo stiffness; too low and the
	// deck lags its sinusoid, too high and it re-approaches the old treadmill
	// (ignoring contact). _fallback_mass_kg is used only if the inertial component
	// cannot be read.
	//
	// _gravity_magnitude is READ but currently UNUSED in PreUpdate:
	// bee_platform.sdf's platform_link already carries <gravity>false</gravity>,
	// so Gazebo never applies gravity to this link, and a compensating force here
	// would cancel something that isn't there -- exactly the bug that was here
	// before (a spurious constant ~m*g force that, combined with the velocity
	// clamp, drove the deck into a large, irregular limit cycle far outside its
	// commanded amplitude/period). Kept as SDF-configurable infrastructure only
	// for the day <gravity> is re-enabled on the link, at which point PreUpdate's
	// AddWorldForce must add this back -- the two must change together.
	double _velocity_tracking_gain{50.0};
	double _fallback_mass_kg{1000.0};
	double _gravity_magnitude{9.8};
	mutable std::optional<double> _cached_mass{};

	std::string _pose_topic{"/platform/pose"};
	gz::transport::Node _transport_node;
	gz::transport::Node::Publisher _pose_publisher;
};
} // namespace custom
