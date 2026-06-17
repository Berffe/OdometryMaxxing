/****************************************************************************
 *
 *   Copyright (c) 2025 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#pragma once

#include <gz/math.hh>
#include <gz/math/Pose3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>

#include <sdf/Element.hh>

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

	gz::math::Vector3d sinusoidalPosition(double time_sec) const;

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
};
} // namespace custom
