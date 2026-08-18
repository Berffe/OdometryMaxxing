/****************************************************************************
 *
 *   BEE_LAND deterministic Gazebo wind command system
 *
 ****************************************************************************/

#include "WindController.hpp"

#include <gz/common/Console.hh>
#include <gz/math/Helpers.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <sstream>

using namespace custom;

namespace
{
constexpr const char *kAxisTag[] = {"axis_x", "axis_y", "axis_z"};
constexpr const char *kAxisName[] = {"x", "y", "z"};
constexpr double kNegligible = 1e-12;
constexpr double kTimeEpsilon = 1e-9;
constexpr double kTruthSchemaVersion = 1.0;

inline int idx(WindAxisId axis)
{
	return static_cast<int>(axis);
}

inline bool isNonZero(double value)
{
	return std::fabs(value) > kNegligible;
}

} // namespace

GZ_ADD_PLUGIN(
	WindController,
	gz::sim::System,
	gz::sim::ISystemConfigure,
	gz::sim::ISystemPreUpdate
)

GZ_ADD_PLUGIN_ALIAS(WindController, "custom::WindController")

// ---------------------------------------------------------------------------
// Configure
// ---------------------------------------------------------------------------

void WindController::Configure(const gz::sim::Entity &entity,
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

	// Clone for mutable sdformat traversal of repeated <component> elements.
	const sdf::ElementPtr cfg = sdf->Clone();

	_windEntity = ecm.EntityByComponents(gz::sim::components::Wind());
	if (_windEntity == gz::sim::kNullEntity) {
		throw std::runtime_error(
			"WindController::Configure: no world wind entity found; add a <wind> element to the world SDF.");
	}

	_enabled = readSdfBool(cfg, "enabled", _enabled);
	_meanVelocity = readSdfVector3(cfg, "mean_velocity", _meanVelocity);
	_maxWindSpeed = readSdfDouble(cfg, "max_wind_speed", _maxWindSpeed);
	_updateRateHz = readSdfDouble(cfg, "update_rate_hz", _updateRateHz);
	_specPublishPeriodSec = readSdfDouble(cfg, "spec_publish_period_sec",
		_specPublishPeriodSec);

	_seed = static_cast<std::uint64_t>(std::max(0, readSdfInt(cfg, "seed", 0)));
	_rng.seed(_seed);

	// Parse order is fixed so seeded synthesis is reproducible across SDF orderings.
	parseAxis(cfg, kAxisTag[idx(WindAxisId::X)], WindAxisId::X);
	parseAxis(cfg, kAxisTag[idx(WindAxisId::Y)], WindAxisId::Y);
	parseAxis(cfg, kAxisTag[idx(WindAxisId::Z)], WindAxisId::Z);

	const std::string defaultWindTopic = "/world/" + _worldName + "/wind";
	_windTopic = readSdfString(cfg, "wind_topic", defaultWindTopic);
	_truthTopic = readSdfString(cfg, "truth_topic", _truthTopic);
	_specTopic = readSdfString(cfg, "spec_topic", _specTopic);

	_windPublisher = _transportNode.Advertise<gz::msgs::Wind>(_windTopic);
	_truthPublisher = _transportNode.Advertise<gz::msgs::Float_V>(_truthTopic);
	_specPublisher = _transportNode.Advertise<gz::msgs::StringMsg>(_specTopic);

	const double speedBound = conservativeSpeedBound();
	if (_maxWindSpeed > 0.0 && speedBound > _maxWindSpeed) {
		gzwarn << "WindController: conservative commanded speed bound "
		       << speedBound << " m/s exceeds max_wind_speed "
		       << _maxWindSpeed << " m/s; the realised command may be clipped.\n";
	}

	gzmsg << describeWind();
}

// ---------------------------------------------------------------------------
// SDF helpers
// ---------------------------------------------------------------------------

double WindController::readSdfDouble(const sdf::ElementPtr &sdf,
		const char *tag, double defaultValue) const
{
	return (sdf && sdf->HasElement(tag)) ? sdf->Get<double>(tag) : defaultValue;
}

int WindController::readSdfInt(const sdf::ElementPtr &sdf,
		const char *tag, int defaultValue) const
{
	return (sdf && sdf->HasElement(tag)) ? sdf->Get<int>(tag) : defaultValue;
}

bool WindController::readSdfBool(const sdf::ElementPtr &sdf,
		const char *tag, bool defaultValue) const
{
	return (sdf && sdf->HasElement(tag)) ? sdf->Get<bool>(tag) : defaultValue;
}

std::string WindController::readSdfString(const sdf::ElementPtr &sdf,
		const char *tag, const std::string &defaultValue) const
{
	return (sdf && sdf->HasElement(tag)) ? sdf->Get<std::string>(tag) : defaultValue;
}

gz::math::Vector3d WindController::readSdfVector3(const sdf::ElementPtr &sdf,
		const char *tag, const gz::math::Vector3d &defaultValue) const
{
	return (sdf && sdf->HasElement(tag))
		? sdf->Get<gz::math::Vector3d>(tag)
		: defaultValue;
}

// ---------------------------------------------------------------------------
// Wind specification
// ---------------------------------------------------------------------------

void WindController::parseAxis(const sdf::ElementPtr &sdf,
		const char *tag, WindAxisId axis)
{
	if (!sdf || !sdf->HasElement(tag)) {
		return;
	}

	const sdf::ElementPtr axisElement = sdf->GetElement(tag);
	parseExplicitComponents(axisElement, axis);

	if (axisElement->HasElement("synthesis")) {
		parseSynthesis(axisElement->GetElement("synthesis"), axis);
	}
}

void WindController::parseExplicitComponents(const sdf::ElementPtr &axisElement,
		WindAxisId axis)
{
	if (!axisElement || !axisElement->HasElement("component")) {
		return;
	}

	for (sdf::ElementPtr component = axisElement->GetElement("component");
	     component;
	     component = component->GetNextElement("component")) {
		WindSineComponent term;
		term.amplitude = readSdfDouble(component, "amplitude", 0.0);
		term.phase = readSdfDouble(component, "phase", 0.0);

		if (component->HasElement("omega")) {
			term.omega = readSdfDouble(component, "omega", 0.0);
		} else {
			term.omega = 2.0 * GZ_PI * readSdfDouble(component, "frequency", 0.0);
		}

		if (isNonZero(term.amplitude) && isNonZero(term.omega)) {
			_components[idx(axis)].push_back(term);
		}
	}
}

void WindController::parseSynthesis(const sdf::ElementPtr &synthesisElement,
		WindAxisId axis)
{
	const std::string mode = readSdfString(synthesisElement, "mode", "uniform");

	if (mode == "uniform") {
		synthesiseUniform(synthesisElement, axis);
	} else {
		gzwarn << "WindController: unknown synthesis mode \"" << mode
		       << "\" on axis " << kAxisName[idx(axis)]
		       << "; no components generated.\n";
	}
}

// ---------------------------------------------------------------------------
// Deterministic synthesis
// ---------------------------------------------------------------------------

double WindController::uniform01()
{
	// Build the double directly from 53 random bits so a seed is independent
	// of the standard library distribution implementation.
	return static_cast<double>(_rng() >> 11) * (1.0 / 9007199254740992.0);
}

double WindController::uniformRange(double lo, double hi)
{
	return lo + (hi - lo) * uniform01();
}

void WindController::synthesiseUniform(const sdf::ElementPtr &synthesisElement,
		WindAxisId axis)
{
	const int count = readSdfInt(synthesisElement, "count", 5);
	const double amplitudeMin = readSdfDouble(synthesisElement,
		"amplitude_min", 0.0);
	const double amplitudeMax = readSdfDouble(synthesisElement,
		"amplitude_max", 0.1);

	double omegaMin = readSdfDouble(synthesisElement, "omega_min", 0.2);
	double omegaMax = readSdfDouble(synthesisElement, "omega_max", 2.0);

	if (synthesisElement->HasElement("frequency_min")) {
		omegaMin = 2.0 * GZ_PI * readSdfDouble(synthesisElement,
			"frequency_min", 0.0);
	}
	if (synthesisElement->HasElement("frequency_max")) {
		omegaMax = 2.0 * GZ_PI * readSdfDouble(synthesisElement,
			"frequency_max", 0.0);
	}

	if (count <= 0 || amplitudeMax < amplitudeMin || omegaMax <= omegaMin ||
	    omegaMin < 0.0) {
		gzwarn << "WindController: invalid uniform synthesis parameters on axis "
		       << kAxisName[idx(axis)] << "; no components generated.\n";
		return;
	}

	for (int i = 0; i < count; ++i) {
		WindSineComponent term;
		term.amplitude = uniformRange(amplitudeMin, amplitudeMax);
		term.omega = uniformRange(omegaMin, omegaMax);
		term.phase = uniformRange(0.0, 2.0 * GZ_PI);

		if (isNonZero(term.amplitude) && isNonZero(term.omega)) {
			_components[idx(axis)].push_back(term);
		}
	}
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

double WindController::axisPerturbation(WindAxisId axis, double timeSec) const
{
	double sum = 0.0;
	for (const WindSineComponent &term : _components[idx(axis)]) {
		sum += term.amplitude * std::sin(term.omega * timeSec + term.phase);
	}
	return sum;
}

gz::math::Vector3d WindController::rawWind(double timeSec) const
{
	return {
		_meanVelocity.X() + axisPerturbation(WindAxisId::X, timeSec),
		_meanVelocity.Y() + axisPerturbation(WindAxisId::Y, timeSec),
		_meanVelocity.Z() + axisPerturbation(WindAxisId::Z, timeSec)
	};
}

gz::math::Vector3d WindController::clampVectorNorm(
		const gz::math::Vector3d &value, double maxNorm) const
{
	if (maxNorm <= 0.0) {
		return value;
	}

	const double norm = value.Length();
	if (norm <= maxNorm || norm <= kNegligible) {
		return value;
	}
	return value * (maxNorm / norm);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

double WindController::boundAxisPerturbation(WindAxisId axis) const
{
	double sum = 0.0;
	for (const WindSineComponent &term : _components[idx(axis)]) {
		sum += std::fabs(term.amplitude);
	}
	return sum;
}

double WindController::boundAxisRate(WindAxisId axis) const
{
	double sum = 0.0;
	for (const WindSineComponent &term : _components[idx(axis)]) {
		sum += std::fabs(term.amplitude * term.omega);
	}
	return sum;
}

double WindController::conservativeSpeedBound() const
{
	const double x = std::fabs(_meanVelocity.X()) + boundAxisPerturbation(WindAxisId::X);
	const double y = std::fabs(_meanVelocity.Y()) + boundAxisPerturbation(WindAxisId::Y);
	const double z = std::fabs(_meanVelocity.Z()) + boundAxisPerturbation(WindAxisId::Z);
	return std::sqrt(x * x + y * y + z * z);
}

std::string WindController::describeWind() const
{
	std::ostringstream os;
	os << std::fixed << std::setprecision(6);
	os << "WindController specification (seed=" << _seed
	   << ", enabled=" << (_enabled ? "true" : "false") << ")\n";
	os << "  mean_velocity_enu=[" << _meanVelocity.X() << ", "
	   << _meanVelocity.Y() << ", " << _meanVelocity.Z() << "] m/s\n";

	for (int a = 0; a < static_cast<int>(WindAxisId::Count); ++a) {
		const WindAxisId axis = static_cast<WindAxisId>(a);
		const auto &list = _components[a];
		const double mean = (axis == WindAxisId::X)
			? _meanVelocity.X()
			: ((axis == WindAxisId::Y) ? _meanVelocity.Y() : _meanVelocity.Z());
		const double ampBound = boundAxisPerturbation(axis);

		os << "  " << kAxisName[a] << ": mean=" << mean << " m/s, "
		   << list.size() << " component(s)\n";
		for (std::size_t i = 0; i < list.size(); ++i) {
			os << "    [" << i << "] amplitude=" << list[i].amplitude << " m/s"
			   << "  omega=" << list[i].omega << " rad/s"
			   << "  (f=" << list[i].omega / (2.0 * GZ_PI) << " Hz)"
			   << "  phase=" << list[i].phase << " rad\n";
		}
		os << "    bounds: command in [" << mean - ampBound << ", "
		   << mean + ampBound << "] m/s"
		   << "  |dW/dt|<=" << boundAxisRate(axis) << " m/s^2\n";
	}

	os << "  conservative |W| bound=" << conservativeSpeedBound() << " m/s\n";
	if (_maxWindSpeed > 0.0) {
		os << "  max_wind_speed=" << _maxWindSpeed << " m/s\n";
	}
	os << "  wind_topic=" << _windTopic << "\n";
	os << "  truth_topic=" << _truthTopic << "\n";
	return os.str();
}

void WindController::writeWindSeed(gz::sim::EntityComponentManager &ecm,
		const gz::math::Vector3d &command)
{
	auto seed = ecm.Component<gz::sim::components::WorldLinearVelocitySeed>(_windEntity);
	if (seed) {
		seed->Data() = command;
	} else {
		ecm.CreateComponent(
			_windEntity,
			gz::sim::components::WorldLinearVelocitySeed(command));
	}
}

void WindController::publishEnableCommand(const gz::math::Vector3d &command)
{
	// WindEffects defaults to enabled, but publish the configured state once per
	// run so <enabled>false</enabled> truly disables the global force system.
	gz::msgs::Wind msg;
	msg.set_enable_wind(_enabled);
	msg.mutable_linear_velocity()->set_x(command.X());
	msg.mutable_linear_velocity()->set_y(command.Y());
	msg.mutable_linear_velocity()->set_z(command.Z());
	_windPublisher.Publish(msg);
	_enableCommandPublished = true;
}

void WindController::publishTruth(double simTimeSec,
		const gz::math::Vector3d &rawCommand,
		const gz::math::Vector3d &command,
		bool clamped)
{
	// Fixed layout:
	// [schema, sequence, sim_time, enabled,
	//  command_x, command_y, command_z, command_norm,
	//  raw_x, raw_y, raw_z, raw_norm, clamped]
	gz::msgs::Float_V msg;
	msg.add_data(static_cast<float>(kTruthSchemaVersion));
	msg.add_data(static_cast<float>(_sequence));
	msg.add_data(static_cast<float>(simTimeSec));
	msg.add_data(_enabled ? 1.0f : 0.0f);
	msg.add_data(static_cast<float>(command.X()));
	msg.add_data(static_cast<float>(command.Y()));
	msg.add_data(static_cast<float>(command.Z()));
	msg.add_data(static_cast<float>(command.Length()));
	msg.add_data(static_cast<float>(rawCommand.X()));
	msg.add_data(static_cast<float>(rawCommand.Y()));
	msg.add_data(static_cast<float>(rawCommand.Z()));
	msg.add_data(static_cast<float>(rawCommand.Length()));
	msg.add_data(clamped ? 1.0f : 0.0f);
	_truthPublisher.Publish(msg);
}

void WindController::publishSpec()
{
	gz::msgs::StringMsg msg;
	msg.set_data(describeWind());
	_specPublisher.Publish(msg);
}

void WindController::resetRuntimeState()
{
	_sequence = 0;
	_lastPublishSimTimeSec = -1e9;
	_lastSpecPublishSimTimeSec = -1e9;
	_enableCommandPublished = false;
}

// ---------------------------------------------------------------------------
// Update loop
// ---------------------------------------------------------------------------

void WindController::PreUpdate(const gz::sim::UpdateInfo &info,
		gz::sim::EntityComponentManager &ecm)
{
	if (info.paused) {
		return;
	}

	const double simTimeSec = std::chrono::duration<double>(info.simTime).count();
	if (_lastObservedSimTimeSec >= 0.0 &&
	    simTimeSec + kTimeEpsilon < _lastObservedSimTimeSec) {
		resetRuntimeState();
	}
	_lastObservedSimTimeSec = simTimeSec;

	const double updatePeriodSec = _updateRateHz > 0.0 ? 1.0 / _updateRateHz : 0.0;
	const bool rateDue = updatePeriodSec <= 0.0 ||
		simTimeSec - _lastPublishSimTimeSec + kTimeEpsilon >= updatePeriodSec;

	if (rateDue) {
		const gz::math::Vector3d rawCommand = _enabled
			? rawWind(simTimeSec)
			: gz::math::Vector3d::Zero;
		const gz::math::Vector3d command = clampVectorNorm(rawCommand, _maxWindSpeed);
		const bool clamped = (command - rawCommand).Length() > kNegligible;

		writeWindSeed(ecm, command);
		if (!_enableCommandPublished) {
			publishEnableCommand(command);
		}
		publishTruth(simTimeSec, rawCommand, command, clamped);
		++_sequence;
		_lastPublishSimTimeSec = simTimeSec;
	}

	if (simTimeSec - _lastSpecPublishSimTimeSec + kTimeEpsilon >=
	    _specPublishPeriodSec) {
		publishSpec();
		_lastSpecPublishSimTimeSec = simTimeSec;
	}
}
