/****************************************************************************
 *
 *   Copyright (c) 2025 PX4 Development Team. All rights reserved.
 *
 ****************************************************************************/

#include "MovingPlatformController.hpp"

#include <gz/common/Console.hh>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace custom;

namespace
{
constexpr const char *kAxisTag[] = {
	"axis_x", "axis_y", "axis_z", "axis_roll", "axis_pitch"
};
constexpr const char *kAxisName[] = {
	"x", "y", "z", "roll", "pitch"
};
inline int idx(AxisId a) { return static_cast<int>(a); }

// PX4 builds with -Werror=float-equal, so components are screened with a
// magnitude threshold rather than an exact comparison against zero. The
// threshold is well below any physically meaningful amplitude (in metres or
// radians) or frequency (in rad/s), so it only rejects entries that were
// omitted from the SDF and defaulted to zero.
constexpr double kNegligible = 1e-12;
inline bool isNonZero(double v) { return std::fabs(v) > kNegligible; }
}

GZ_ADD_PLUGIN(
	OscillatingPlatformController,
	gz::sim::System,
	gz::sim::ISystemConfigure,
	gz::sim::ISystemPreUpdate,
	gz::sim::ISystemPostUpdate
)

// ---------------------------------------------------------------------------
// Configure
// ---------------------------------------------------------------------------

void OscillatingPlatformController::Configure(const gz::sim::Entity &entity,
		const std::shared_ptr<const sdf::Element> &sdf,
		gz::sim::EntityComponentManager &ecm,
		gz::sim::EventManager &eventMgr)
{
	(void)eventMgr;

	// Clone to obtain a mutable ElementPtr; the sdformat traversal API
	// (GetElement / GetNextElement) is not available on a const element.
	const sdf::ElementPtr cfg = sdf->Clone();

	_model = gz::sim::Model(entity);

	const std::string link_name = readSdfString(cfg, "link_name", "platform_link");
	const gz::sim::Entity link_entity = _model.LinkByName(ecm, link_name);

	if (link_entity == gz::sim::kNullEntity) {
		throw std::runtime_error("OscillatingPlatformController::Configure: link \"" + link_name + "\" was not found.");
	}

	_link = gz::sim::Link(link_entity);
	_link.EnableVelocityChecks(ecm, true);

	const auto link_pose = _link.WorldPose(ecm);
	_initial_pose = link_pose.has_value() ? *link_pose : gz::sim::worldPose(entity, ecm);

	// Seed first: every synthesis block below draws from this one stream, so
	// the axis parse order is part of the reproducibility contract. Axes are
	// always parsed X, Y, Z, Roll, Pitch regardless of SDF ordering.
	_seed = static_cast<std::uint64_t>(readSdfInt(cfg, "seed", 0));
	_rng.seed(_seed);

	parseAxis(cfg, kAxisTag[idx(AxisId::X)],     AxisId::X);
	parseAxis(cfg, kAxisTag[idx(AxisId::Y)],     AxisId::Y);
	parseAxis(cfg, kAxisTag[idx(AxisId::Z)],     AxisId::Z);
	parseAxis(cfg, kAxisTag[idx(AxisId::Roll)],  AxisId::Roll);
	parseAxis(cfg, kAxisTag[idx(AxisId::Pitch)], AxisId::Pitch);

	// Legacy flat tags, applied only where the new block left an axis empty.
	if (_components[idx(AxisId::X)].empty()) {
		parseLegacyAxis(cfg, "x_amplitude", "x_frequency", "x_phase", AxisId::X);
	}

	if (_components[idx(AxisId::Y)].empty()) {
		parseLegacyAxis(cfg, "y_amplitude", "y_frequency", "y_phase", AxisId::Y);
	}

	if (_components[idx(AxisId::Z)].empty()) {
		parseLegacyAxis(cfg, "z_amplitude", "z_frequency", "z_phase", AxisId::Z);
	}

	_position_gain = readSdfDouble(cfg, "position_gain", _position_gain);
	_max_linear_velocity = readSdfDouble(cfg, "max_linear_velocity", _max_linear_velocity);

	_angular_enabled = readSdfBool(cfg, "angular_enabled", _angular_enabled);
	_angular_gain = readSdfDouble(cfg, "angular_gain", _angular_gain);
	_max_angular_velocity = readSdfDouble(cfg, "max_angular_velocity", _max_angular_velocity);

	_pose_topic = readSdfString(cfg, "pose_topic", _pose_topic);
	_spec_topic = readSdfString(cfg, "spec_topic", _spec_topic);
	_pose_publisher = _transport_node.Advertise<gz::msgs::Pose>(_pose_topic);
	_spec_publisher = _transport_node.Advertise<gz::msgs::StringMsg>(_spec_topic);

	// Warn rather than fail: a clipped disturbance is still a valid run, but a
	// silently clipped one invalidates any claim about what was tested.
	const double v_bound = std::sqrt(
			boundVelocity(AxisId::X) * boundVelocity(AxisId::X)
			+ boundVelocity(AxisId::Y) * boundVelocity(AxisId::Y)
			+ boundVelocity(AxisId::Z) * boundVelocity(AxisId::Z));

	if (_max_linear_velocity > 0.0 && v_bound > _max_linear_velocity) {
		gzwarn << "OscillatingPlatformController: worst-case commanded speed "
		       << v_bound << " m/s exceeds max_linear_velocity "
		       << _max_linear_velocity << " m/s; the realised motion may be clipped.\n";
	}

	gzmsg << describeMotion();
}

// ---------------------------------------------------------------------------
// SDF helpers
// ---------------------------------------------------------------------------

double OscillatingPlatformController::readSdfDouble(const sdf::ElementPtr &sdf,
		const char *tag, double default_value) const
{
	return (sdf && sdf->HasElement(tag)) ? sdf->Get<double>(tag) : default_value;
}

int OscillatingPlatformController::readSdfInt(const sdf::ElementPtr &sdf,
		const char *tag, int default_value) const
{
	return (sdf && sdf->HasElement(tag)) ? sdf->Get<int>(tag) : default_value;
}

bool OscillatingPlatformController::readSdfBool(const sdf::ElementPtr &sdf,
		const char *tag, bool default_value) const
{
	return (sdf && sdf->HasElement(tag)) ? sdf->Get<bool>(tag) : default_value;
}

std::string OscillatingPlatformController::readSdfString(const sdf::ElementPtr &sdf,
		const char *tag, const std::string &default_value) const
{
	return (sdf && sdf->HasElement(tag)) ? sdf->Get<std::string>(tag) : default_value;
}

// ---------------------------------------------------------------------------
// Motion specification
// ---------------------------------------------------------------------------

void OscillatingPlatformController::parseAxis(const sdf::ElementPtr &sdf,
		const char *tag, AxisId axis)
{
	if (!sdf->HasElement(tag)) {
		return;
	}

	const sdf::ElementPtr axis_elem = sdf->GetElement(tag);
	parseExplicitComponents(axis_elem, axis);

	if (axis_elem->HasElement("synthesis")) {
		parseSynthesis(axis_elem->GetElement("synthesis"), axis);
	}
}

void OscillatingPlatformController::parseExplicitComponents(const sdf::ElementPtr &axis_elem,
		AxisId axis)
{
	if (!axis_elem->HasElement("component")) {
		return;
	}

	for (sdf::ElementPtr c = axis_elem->GetElement("component"); c;
	     c = c->GetNextElement("component")) {

		SineComponent comp;
		comp.amplitude = readSdfDouble(c, "amplitude", 0.0);
		comp.phase     = readSdfDouble(c, "phase", 0.0);

		// Accept either <frequency> in Hz (matching the original flat tags)
		// or <omega> in rad/s (matching Herisse's parameterisation), so the
		// paper's numbers can be transcribed in whichever unit they appear.
		if (c->HasElement("omega")) {
			comp.omega = readSdfDouble(c, "omega", 0.0);

		} else {
			comp.omega = 2.0 * GZ_PI * readSdfDouble(c, "frequency", 0.0);
		}

		if (isNonZero(comp.amplitude) && isNonZero(comp.omega)) {
			_components[idx(axis)].push_back(comp);
		}
	}
}

void OscillatingPlatformController::parseSynthesis(const sdf::ElementPtr &synth_elem,
		AxisId axis)
{
	const std::string mode = readSdfString(synth_elem, "mode", "uniform");

	if (mode == "uniform") {
		synthesiseUniform(synth_elem, axis);

	} else if (mode == "pierson_moskowitz" || mode == "pm") {
		synthesisePiersonMoskowitz(synth_elem, axis);

	} else {
		gzwarn << "OscillatingPlatformController: unknown synthesis mode \""
		       << mode << "\" on axis " << kAxisName[idx(axis)]
		       << "; no components generated.\n";
	}
}

void OscillatingPlatformController::parseLegacyAxis(const sdf::ElementPtr &sdf,
		const char *amp_tag, const char *freq_tag, const char *phase_tag, AxisId axis)
{
	const double amplitude = readSdfDouble(sdf, amp_tag, 0.0);
	const double frequency = readSdfDouble(sdf, freq_tag, 0.0);

	if (!isNonZero(amplitude) || !isNonZero(frequency)) {
		return;
	}

	SineComponent comp;
	comp.amplitude = amplitude;
	comp.omega = 2.0 * GZ_PI * frequency;
	comp.phase = readSdfDouble(sdf, phase_tag, 0.0);
	_components[idx(axis)].push_back(comp);
}

// ---------------------------------------------------------------------------
// Deterministic sampling
// ---------------------------------------------------------------------------

double OscillatingPlatformController::uniform01()
{
	// 53 significant bits from the 64-bit engine output. Hand-rolled so the
	// sequence does not depend on the standard library implementation.
	return static_cast<double>(_rng() >> 11) * (1.0 / 9007199254740992.0);
}

double OscillatingPlatformController::uniformRange(double lo, double hi)
{
	return lo + (hi - lo) * uniform01();
}

void OscillatingPlatformController::synthesiseUniform(const sdf::ElementPtr &synth_elem,
		AxisId axis)
{
	const int n = readSdfInt(synth_elem, "count", 7);

	const double a_min = readSdfDouble(synth_elem, "amplitude_min", 0.0);
	const double a_max = readSdfDouble(synth_elem, "amplitude_max", 0.1);

	// Frequency bounds in rad/s by default (Herisse's convention); <frequency_min>
	// / <frequency_max> in Hz are accepted as an alternative.
	double w_min = readSdfDouble(synth_elem, "omega_min", 1.0);
	double w_max = readSdfDouble(synth_elem, "omega_max", 6.0);

	if (synth_elem->HasElement("frequency_min")) {
		w_min = 2.0 * GZ_PI * readSdfDouble(synth_elem, "frequency_min", 0.0);
	}

	if (synth_elem->HasElement("frequency_max")) {
		w_max = 2.0 * GZ_PI * readSdfDouble(synth_elem, "frequency_max", 0.0);
	}

	for (int i = 0; i < n; ++i) {
		SineComponent comp;
		comp.amplitude = uniformRange(a_min, a_max);
		comp.omega = uniformRange(w_min, w_max);
		comp.phase = uniformRange(0.0, 2.0 * GZ_PI);
		_components[idx(axis)].push_back(comp);
	}
}

void OscillatingPlatformController::synthesisePiersonMoskowitz(const sdf::ElementPtr &synth_elem,
		AxisId axis)
{
	const int n = readSdfInt(synth_elem, "count", 24);

	// Significant height and peak frequency define the spectrum;
	//   S(f) = (5/16) Hs^2 fp^4 f^-5 exp(-1.25 (fp/f)^4)
	// Amplitudes follow as a_i = sqrt(2 S(f_i) df), phases are random. Only
	// the phases are stochastic here: this is one realisation of a specified
	// sea state, not a bounded family.
	const double hs = readSdfDouble(synth_elem, "significant_height", 0.4);
	const double fp = readSdfDouble(synth_elem, "peak_frequency", 0.2);
	const double f_lo = readSdfDouble(synth_elem, "frequency_min", 0.5 * fp);
	const double f_hi = readSdfDouble(synth_elem, "frequency_max", 3.0 * fp);

	if (n <= 0 || fp <= 0.0 || f_hi <= f_lo) {
		gzwarn << "OscillatingPlatformController: invalid Pierson-Moskowitz parameters on axis "
		       << kAxisName[idx(axis)] << "; no components generated.\n";
		return;
	}

	const double df = (f_hi - f_lo) / static_cast<double>(n);

	for (int i = 0; i < n; ++i) {
		// Mid-bin frequency, so no component sits exactly on the band edge.
		const double f = f_lo + (static_cast<double>(i) + 0.5) * df;
		const double r = fp / f;
		const double s = (5.0 / 16.0) * hs * hs * std::pow(fp, 4.0)
				 * std::pow(f, -5.0) * std::exp(-1.25 * std::pow(r, 4.0));

		SineComponent comp;
		comp.amplitude = std::sqrt(2.0 * s * df);
		comp.omega = 2.0 * GZ_PI * f;
		comp.phase = uniformRange(0.0, 2.0 * GZ_PI);

		if (comp.amplitude > 0.0) {
			_components[idx(axis)].push_back(comp);
		}
	}
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

double OscillatingPlatformController::axisPosition(AxisId axis, double t) const
{
	double sum = 0.0;

	for (const SineComponent &c : _components[idx(axis)]) {
		sum += c.amplitude * std::sin(c.omega * t + c.phase);
	}

	return sum;
}

double OscillatingPlatformController::axisVelocity(AxisId axis, double t) const
{
	double sum = 0.0;

	for (const SineComponent &c : _components[idx(axis)]) {
		sum += c.amplitude * c.omega * std::cos(c.omega * t + c.phase);
	}

	return sum;
}

gz::math::Vector3d OscillatingPlatformController::sinusoidalPosition(double time_sec) const
{
	return {_initial_pose.Pos().X() + axisPosition(AxisId::X, time_sec),
		_initial_pose.Pos().Y() + axisPosition(AxisId::Y, time_sec),
		_initial_pose.Pos().Z() + axisPosition(AxisId::Z, time_sec)};
}

gz::math::Vector3d OscillatingPlatformController::sinusoidalVelocity(double time_sec) const
{
	return {axisVelocity(AxisId::X, time_sec),
		axisVelocity(AxisId::Y, time_sec),
		axisVelocity(AxisId::Z, time_sec)};
}

gz::math::Vector3d OscillatingPlatformController::clampVectorNorm(const gz::math::Vector3d &value,
		double max_norm) const
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

gz::math::Quaterniond OscillatingPlatformController::angularTarget(double time_sec) const
{
	const double roll  = axisPosition(AxisId::Roll,  time_sec);
	const double pitch = axisPosition(AxisId::Pitch, time_sec);
	return gz::math::Quaterniond(roll, pitch, 0.0);
}

gz::math::Vector3d OscillatingPlatformController::angularVelocityCommand(double time_sec,
		const gz::math::Quaterniond &current) const
{
	// Feedforward from the analytic rates, plus proportional correction on the
	// attitude error. For the small tilt angles a deck exhibits, the roll/pitch
	// rates are close enough to the body angular velocity that no Euler-rate
	// transformation is warranted; the P term absorbs the residual.
	const gz::math::Vector3d feedforward(axisVelocity(AxisId::Roll, time_sec),
					     axisVelocity(AxisId::Pitch, time_sec),
					     0.0);

	const gz::math::Quaterniond target = angularTarget(time_sec);
	gz::math::Quaterniond error = target * current.Inverse();
	error.Normalize();

	// Shortest-arc: flip if the scalar part is negative.
	if (error.W() < 0.0) {
		error.Set(-error.W(), -error.X(), -error.Y(), -error.Z());
	}

	const gz::math::Vector3d correction(2.0 * error.X(), 2.0 * error.Y(), 2.0 * error.Z());

	return clampVectorNorm(feedforward + _angular_gain * correction, _max_angular_velocity);
}

// ---------------------------------------------------------------------------
// Update loop
// ---------------------------------------------------------------------------

void OscillatingPlatformController::PreUpdate(const gz::sim::UpdateInfo &info,
		gz::sim::EntityComponentManager &ecm)
{
	if (info.paused || _link.Entity() == gz::sim::kNullEntity) {
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

	_link.SetLinearVelocity(ecm, commanded_velocity);

	if (_angular_enabled) {
		_link.SetAngularVelocity(ecm, angularVelocityCommand(t, current_pose->Rot()));

	} else {
		_link.SetAngularVelocity(ecm, gz::math::Vector3d::Zero);
	}
}

void OscillatingPlatformController::PostUpdate(const gz::sim::UpdateInfo &info,
		const gz::sim::EntityComponentManager &ecm)
{
	if (info.paused || _link.Entity() == gz::sim::kNullEntity) {
		return;
	}

	const auto pose = _link.WorldPose(ecm);

	if (pose.has_value()) {
		publishPose(*pose);
	}

	const double t = std::chrono::duration<double>(info.simTime).count();

	if (t - _last_spec_publish_sec >= _spec_publish_period_sec) {
		_last_spec_publish_sec = t;
		publishSpec();
	}
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

double OscillatingPlatformController::boundDisplacement(AxisId axis) const
{
	double sum = 0.0;

	for (const SineComponent &c : _components[idx(axis)]) {
		sum += std::fabs(c.amplitude);
	}

	return sum;
}

double OscillatingPlatformController::boundVelocity(AxisId axis) const
{
	double sum = 0.0;

	for (const SineComponent &c : _components[idx(axis)]) {
		sum += std::fabs(c.amplitude * c.omega);
	}

	return sum;
}

double OscillatingPlatformController::boundAcceleration(AxisId axis) const
{
	double sum = 0.0;

	for (const SineComponent &c : _components[idx(axis)]) {
		sum += std::fabs(c.amplitude * c.omega * c.omega);
	}

	return sum;
}

std::string OscillatingPlatformController::describeMotion() const
{
	std::ostringstream os;
	os << std::fixed << std::setprecision(6);
	os << "OscillatingPlatformController motion specification (seed=" << _seed << ")\n";

	for (int a = 0; a < static_cast<int>(AxisId::Count); ++a) {
		const AxisId axis = static_cast<AxisId>(a);
		const auto &list = _components[a];

		const bool angular = (axis == AxisId::Roll || axis == AxisId::Pitch);
		const char *unit = angular ? "rad" : "m";

		if (list.empty()) {
			os << "  " << kAxisName[a] << ": static\n";
			continue;
		}

		if (angular && !_angular_enabled) {
			os << "  " << kAxisName[a] << ": " << list.size()
			   << " component(s) DEFINED BUT DISABLED (angular_enabled=false)\n";

		} else {
			os << "  " << kAxisName[a] << ": " << list.size() << " component(s)\n";
		}

		for (std::size_t i = 0; i < list.size(); ++i) {
			os << "    [" << i << "] amplitude=" << list[i].amplitude << " " << unit
			   << "  omega=" << list[i].omega << " rad/s"
			   << "  (f=" << list[i].omega / (2.0 * GZ_PI) << " Hz)"
			   << "  phase=" << list[i].phase << " rad\n";
		}

		os << "    bounds: |disp|<=" << boundDisplacement(axis) << " " << unit
		   << "  |vel|<=" << boundVelocity(axis) << " " << unit << "/s"
		   << "  |acc|<=" << boundAcceleration(axis) << " " << unit << "/s^2\n";
	}

	return os.str();
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

void OscillatingPlatformController::publishSpec()
{
	gz::msgs::StringMsg msg;
	msg.set_data(describeMotion());
	_spec_publisher.Publish(msg);
}
