/****************************************************************************
 *
 *   Copyright (c) 2025 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#include "MovingPlatformController.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

using namespace custom;

GZ_ADD_PLUGIN(
	OscillatingPlatformController,
	gz::sim::System,
	gz::sim::ISystemPreUpdate,
	gz::sim::ISystemConfigure
)

void OscillatingPlatformController::Configure(const gz::sim::Entity &entity,
		const std::shared_ptr<const sdf::Element> &sdf,
		gz::sim::EntityComponentManager &ecm,
		gz::sim::EventManager &eventMgr)
{
	(void)eventMgr;

	_model = gz::sim::Model(entity);

	const std::string link_name = sdf->HasElement("link_name") ? sdf->Get<std::string>("link_name") : "platform_link";
	const gz::sim::Entity link_entity = _model.LinkByName(ecm, link_name);

	if (link_entity == gz::sim::kNullEntity) {
		throw std::runtime_error("OscillatingPlatformController::Configure: link \"" + link_name + "\" was not found.");
	}

	_initial_pose = gz::sim::worldPose(entity, ecm);

	_x_amplitude = readSdfDouble(sdf, "x_amplitude", _x_amplitude);
	_x_frequency = readSdfDouble(sdf, "x_frequency", _x_frequency);
	_x_phase = readSdfDouble(sdf, "x_phase", _x_phase);

	_y_amplitude = readSdfDouble(sdf, "y_amplitude", _y_amplitude);
	_y_frequency = readSdfDouble(sdf, "y_frequency", _y_frequency);
	_y_phase = readSdfDouble(sdf, "y_phase", _y_phase);

	_z_amplitude = readSdfDouble(sdf, "z_amplitude", _z_amplitude);
	_z_frequency = readSdfDouble(sdf, "z_frequency", _z_frequency);
	_z_phase = readSdfDouble(sdf, "z_phase", _z_phase);

	// Optional <pose_topic> SDF tag; defaults to /platform/pose (see the
	// header's comment on publishPose for why this publisher exists at all).
	_pose_topic = readSdfString(sdf, "pose_topic", _pose_topic);
	_pose_publisher = _transport_node.Advertise<gz::msgs::Pose>(_pose_topic);
}

void OscillatingPlatformController::PreUpdate(const gz::sim::UpdateInfo &info,
		gz::sim::EntityComponentManager &ecm)
{
	if (info.paused) {
		return;
	}

	const double t = std::chrono::duration<double>(info.simTime).count();

	gz::math::Pose3d target_pose = _initial_pose;
	target_pose.Pos() = sinusoidalPosition(t);

	_model.SetWorldPoseCmd(ecm, target_pose);
	publishPose(target_pose);
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