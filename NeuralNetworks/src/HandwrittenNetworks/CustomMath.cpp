#include "Math.h"

#include <random>

namespace Math
{
	std::random_device g_RandomDevice;
	std::mt19937 g_NumberGenerator(g_RandomDevice());
	std::uniform_real_distribution<float> g_FloatDistribution(0, 1);

	float RandomFloat()
	{
		return g_FloatDistribution(g_NumberGenerator);
	}

	float Sigmoid(const float X)
	{
		// f(x) = 1 / ( 1 + exp(-x))
		return 1.0f / (1.0f + std::exp(-X));
	}

	double Sigmoid(const double X)
	{
		// f(x) = 1 / ( 1 + exp(-x))
		return 1.0f / (1.0f + std::exp(-X));
	}
}
