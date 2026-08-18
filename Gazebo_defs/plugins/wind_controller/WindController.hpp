/****************************************************************************
 *
 *   BEE_LAND deterministic Gazebo wind command system
 *
 ****************************************************************************/

#pragma once

#include <gz/math/Vector3.hh>
#include <gz/msgs/float_v.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/msgs/wind.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/LinearVelocitySeed.hh>
#include <gz/sim/components/Wind.hh>
#include <gz/sim/System.hh>
#include <gz/sim/World.hh>
#include <gz/transport/Node.hh>

#include <sdf/Element.hh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace custom
{

// One Cartesian wind component: amplitude * sin(omega * t + phase).
// Amplitude is in m/s, omega in rad/s and phase in rad.
struct WindSineComponent
{
	double amplitude{0.0};
	double omega{0.0};
	double phase{0.0};
};

enum class WindAxisId : int { X = 0, Y = 1, Z = 2, Count = 3 };

class WindController:
	public gz::sim::System,
	public gz::sim::ISystemConfigure,
	public gz::sim::ISystemPreUpdate
{
public:
	void Configure(const gz::sim::Entity &entity,
			const std::shared_ptr<const sdf::Element> &sdf,
			gz::sim::EntityComponentManager &ecm,
			gz::sim::EventManager &eventMgr) override;

	void PreUpdate(const gz::sim::UpdateInfo &info,
			gz::sim::EntityComponentManager &ecm) final;

private:
	// ---- SDF helpers -------------------------------------------------------
	double readSdfDouble(const sdf::ElementPtr &sdf, const char *tag,
			double defaultValue) const;
	int readSdfInt(const sdf::ElementPtr &sdf, const char *tag,
			int defaultValue) const;
	bool readSdfBool(const sdf::ElementPtr &sdf, const char *tag,
			bool defaultValue) const;
	std::string readSdfString(const sdf::ElementPtr &sdf, const char *tag,
			const std::string &defaultValue) const;
	gz::math::Vector3d readSdfVector3(const sdf::ElementPtr &sdf, const char *tag,
			const gz::math::Vector3d &defaultValue) const;

	// ---- Wind specification ----------------------------------------------
	void parseAxis(const sdf::ElementPtr &sdf, const char *tag, WindAxisId axis);
	void parseExplicitComponents(const sdf::ElementPtr &axisElement,
			WindAxisId axis);
	void parseSynthesis(const sdf::ElementPtr &synthesisElement,
			WindAxisId axis);
	void synthesiseUniform(const sdf::ElementPtr &synthesisElement,
			WindAxisId axis);

	// ---- Deterministic sampling -------------------------------------------
	double uniform01();
	double uniformRange(double lo, double hi);

	// ---- Evaluation --------------------------------------------------------
	double axisPerturbation(WindAxisId axis, double timeSec) const;
	gz::math::Vector3d rawWind(double timeSec) const;
	gz::math::Vector3d clampVectorNorm(const gz::math::Vector3d &value,
			double maxNorm) const;

	// ---- Diagnostics -------------------------------------------------------
	double boundAxisPerturbation(WindAxisId axis) const;
	double boundAxisRate(WindAxisId axis) const;
	double conservativeSpeedBound() const;
	std::string describeWind() const;
	void writeWindSeed(gz::sim::EntityComponentManager &ecm,
			const gz::math::Vector3d &command);
	void publishEnableCommand(const gz::math::Vector3d &command);
	void publishTruth(double simTimeSec, const gz::math::Vector3d &rawCommand,
			const gz::math::Vector3d &command, bool clamped);
	void publishSpec();
	void resetRuntimeState();

	// ---- State -------------------------------------------------------------
	gz::sim::Entity _worldEntity{gz::sim::kNullEntity};
	gz::sim::Entity _windEntity{gz::sim::kNullEntity};
	std::string _worldName{"bee_platform"};

	bool _enabled{true};
	gz::math::Vector3d _meanVelocity{0.0, 0.0, 0.0};
	std::vector<WindSineComponent> _components[static_cast<int>(WindAxisId::Count)];

	std::uint64_t _seed{0};
	std::mt19937_64 _rng{0};

	// If <= 0, no norm clamp is applied.
	double _maxWindSpeed{0.0};
	// If <= 0, publish every simulation step.
	double _updateRateHz{0.0};
	double _specPublishPeriodSec{2.0};

	std::string _windTopic;
	std::string _truthTopic{"/bee_land/wind_cmd"};
	std::string _specTopic{"/bee_land/wind_spec"};

	gz::transport::Node _transportNode;
	gz::transport::Node::Publisher _windPublisher;
	gz::transport::Node::Publisher _truthPublisher;
	gz::transport::Node::Publisher _specPublisher;

	std::uint64_t _sequence{0};
	double _lastObservedSimTimeSec{-1.0};
	double _lastPublishSimTimeSec{-1e9};
	double _lastSpecPublishSimTimeSec{-1e9};
	bool _enableCommandPublished{false};
};

} // namespace custom
