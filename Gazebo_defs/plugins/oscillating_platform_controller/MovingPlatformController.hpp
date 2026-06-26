/****************************************************************************
 *
 *   Copyright (c) 2025 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#pragma once

#include <gz/math.hh>
#include <gz/math/Pose3.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
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
	public gz::sim::ISystemPreUpdate,
	public gz::sim::ISystemConfigure
{
public:
	void Configure(const gz::sim::Entity &entity,
			const std::shared_ptr<const sdf::Element> &sdf,
			gz::sim::EntityComponentManager &ecm,
			gz::sim::EventManager &eventMgr) override;

	void PreUpdate(const gz::sim::UpdateInfo &info,
			gz::sim::EntityComponentManager &ecm) final;

private:
	double readSdfDouble(const std::shared_ptr<const sdf::Element> &sdf,
				const char *tag,
				double default_value) const;

	std::string readSdfString(const std::shared_ptr<const sdf::Element> &sdf,
				const char *tag,
				const std::string &default_value) const;

	gz::math::Vector3d sinusoidalPosition(double time_sec) const;

	// Publishes the SAME pose this tick set via SetWorldPoseCmd, as a plain
	// gz::msgs::Pose, on our own dedicated topic. Exists because gz-sim's
	// built-in pose broadcasting (SceneBroadcaster's dynamic_pose/info and
	// pose/info) was found NOT to be a reliable source for this entity:
	// dynamic_pose/info excludes anything tagged <static>true</static>
	// regardless of what a plugin does to its pose, and pose/info's
	// gz.msgs.Pose_V -> ros_gz_bridge tf2_msgs/msg/TFMessage conversion was
	// confirmed (via the consuming ROS node logging every child_frame_id it
	// received) to leave every entity's name empty, making it impossible to
	// pick this entity back out on the ROS side. Publishing our own
	// single-entity topic sidesteps both problems and lets the ROS side use
	// the much simpler gz.msgs.Pose -> geometry_msgs/msg/Pose bridge instead
	// of TFMessage, with no name-matching needed at all.
	void publishPose(const gz::math::Pose3d &pose);

	gz::sim::Model _model{gz::sim::kNullEntity};
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

	std::string _pose_topic{"/platform/pose"};
	gz::transport::Node _transport_node;
	gz::transport::Node::Publisher _pose_publisher;
};
} // namespace custom