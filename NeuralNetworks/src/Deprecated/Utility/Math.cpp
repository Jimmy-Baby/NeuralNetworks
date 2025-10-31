// Precompiled headers
#include "Pch.h"

#include "Math.h"

namespace Math
{
	float RandomFloat()
	{
		std::random_device randomDevice;
		std::mt19937 numberGenerator(randomDevice());
		std::uniform_real_distribution<float> floatDistribution(0, 1);
		return floatDistribution(numberGenerator);
	}

	float Sigmoid(const float x)
	{
		// f(x) = 1 / ( 1 + exp(-x))
		return 1.0f / (1.0f + std::exp(-x));
	}

	double Sigmoid(const double x)
	{
		// f(x) = 1 / ( 1 + exp(-x))
		return 1.0 / (1.0 + std::exp(-x));
	}
}
