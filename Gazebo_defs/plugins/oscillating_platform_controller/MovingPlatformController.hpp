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

	std::string _pose_topic{"/platform/pose"};
	gz::transport::Node _transport_node;
	gz::transport::Node::Publisher _pose_publisher;
};
} // namespace custom
