/****************************************************************************
 *
 *   Copyright (c) 2025 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#pragma once

#include <gz/math.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Quaternion.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/transport/Node.hh>

#include <sdf/Element.hh>

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace custom
{

// One sinusoidal term: amplitude * sin(omega * t + phase).
// Every motion mode this plugin supports -- a single sinusoid, a hand-written
// composition, a Herisse-style bounded random family, or a random-phase
// spectral realisation -- reduces to a list of these. The integrator does not
// need to know which sampler produced them, which is why position and velocity
// stay analytic in all cases and the velocity feedforward keeps working.
struct SineComponent {
	double amplitude{0.0};   // metres, or radians for angular axes
	double omega{0.0};       // rad/s
	double phase{0.0};       // rad
};

// Axis identifiers, used only to key the SDF element names.
enum class AxisId : int { X = 0, Y = 1, Z = 2, Roll = 3, Pitch = 4, Count = 5 };

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
	// ---- SDF helpers -------------------------------------------------------
	double readSdfDouble(const sdf::ElementPtr &sdf, const char *tag, double default_value) const;
	int readSdfInt(const sdf::ElementPtr &sdf, const char *tag, int default_value) const;
	bool readSdfBool(const sdf::ElementPtr &sdf, const char *tag, bool default_value) const;
	std::string readSdfString(const sdf::ElementPtr &sdf, const char *tag,
				const std::string &default_value) const;

	// ---- Motion specification ---------------------------------------------
	// Parses one <axis_x> / <axis_y> / <axis_z> / <axis_roll> / <axis_pitch>
	// block. Each may contain any number of explicit <component> children and
	// at most one <synthesis> block; both are appended to the same list.
	void parseAxis(const sdf::ElementPtr &sdf, const char *tag, AxisId axis);
	void parseExplicitComponents(const sdf::ElementPtr &axis_elem, AxisId axis);
	void parseSynthesis(const sdf::ElementPtr &synth_elem, AxisId axis);

	// Backward compatibility: the original flat <x_amplitude>/<x_frequency>/
	// <x_phase> tags are read as a single component when no <axis_*> block is
	// present for that axis, so existing world files keep working unchanged.
	void parseLegacyAxis(const sdf::ElementPtr &sdf, const char *amp_tag,
				const char *freq_tag, const char *phase_tag, AxisId axis);

	// ---- Deterministic sampling -------------------------------------------
	// std::mt19937_64 is specified by the standard, but the distribution
	// classes are NOT -- std::uniform_real_distribution may produce different
	// sequences on different toolchains from the same seed. Uniforms are
	// therefore built by hand so that a seed reproduces the same disturbance
	// on any machine, which is the whole point of recording it in a paper.
	double uniform01();
	double uniformRange(double lo, double hi);

	// Herisse et al. (2012), Fig. 8: n components with amplitude, frequency and
	// phase each drawn uniformly from a bounded interval. The bounds make the
	// worst-case acceleration analytically computable (sum a_i * omega_i^2),
	// which is what a gain-floor condition needs. This is a bounded-family
	// robustness test, not a realistic sea.
	void synthesiseUniform(const sdf::ElementPtr &synth_elem, AxisId axis);

	// Random-phase spectral synthesis: amplitudes and frequencies fixed by a
	// Pierson-Moskowitz spectrum, only the phases are random. This is a
	// realisation of one specified sea state rather than a bounded family.
	void synthesisePiersonMoskowitz(const sdf::ElementPtr &synth_elem, AxisId axis);

	// ---- Evaluation --------------------------------------------------------
	double axisPosition(AxisId axis, double t) const;
	double axisVelocity(AxisId axis, double t) const;

	gz::math::Vector3d sinusoidalPosition(double time_sec) const;
	gz::math::Vector3d sinusoidalVelocity(double time_sec) const;
	gz::math::Vector3d clampVectorNorm(const gz::math::Vector3d &value, double max_norm) const;

	// Angular target and its rate. Yaw is always zero: a rotating deck would
	// change the visual target's appearance in ways the vision pipeline does
	// not currently model, and it is not needed for the sea-state cases.
	gz::math::Quaterniond angularTarget(double time_sec) const;
	gz::math::Vector3d angularVelocityCommand(double time_sec,
				const gz::math::Quaterniond &current) const;

	// ---- Diagnostics -------------------------------------------------------
	// Worst-case bounds over the realised component list. sum(a_i) bounds
	// displacement, sum(a_i * w_i) bounds velocity, sum(a_i * w_i^2) bounds
	// acceleration. The last is the quantity a Herisse-style gain condition
	// needs, and the first two tell you whether max_linear_velocity will clip.
	double boundDisplacement(AxisId axis) const;
	double boundVelocity(AxisId axis) const;
	double boundAcceleration(AxisId axis) const;

	// Serialises the realised (amplitude, omega, phase) list. Logged once at
	// startup and republished periodically: reproducing a run from a seed
	// requires the same binary, but the explicit component list can be pasted
	// straight back into an <axis_*> block by anyone, forever.
	std::string describeMotion() const;
	void publishPose(const gz::math::Pose3d &pose);
	void publishSpec();

	// ---- State -------------------------------------------------------------
	gz::sim::Model _model{gz::sim::kNullEntity};
	gz::sim::Link _link{gz::sim::kNullEntity};
	gz::math::Pose3d _initial_pose{0., 0., 0., 0., 0., 0.};

	std::vector<SineComponent> _components[static_cast<int>(AxisId::Count)];

	// Velocity-driven tracking. commanded_velocity = feedforward + gain * error.
	double _position_gain{8.0};
	double _max_linear_velocity{5.0};

	// Angular motion is plumbed through but inert until <angular_enabled> is
	// set true. With it false the link is held level exactly as before.
	bool _angular_enabled{false};
	double _angular_gain{8.0};
	double _max_angular_velocity{2.0};

	std::uint64_t _seed{0};
	std::mt19937_64 _rng{0};

	std::string _pose_topic{"/platform/pose"};
	std::string _spec_topic{"/platform/motion_spec"};
	gz::transport::Node _transport_node;
	gz::transport::Node::Publisher _pose_publisher;
	gz::transport::Node::Publisher _spec_publisher;
	double _last_spec_publish_sec{-1e9};
	double _spec_publish_period_sec{2.0};
};

} // namespace custom
