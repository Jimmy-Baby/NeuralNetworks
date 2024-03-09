#include <random>

#include "Random.h"

namespace Random
{
	void RandomScalarFill(TScalar* Array, const size_t ArraySize, const TScalar& Mean, const TScalar& StandardDeviation)
	{
		constexpr double TWO_PI = 6.283185307179586476925286766559;

		static std::random_device RandomDevice;
		static std::mt19937 RandomEngine(RandomDevice());
		static std::uniform_real_distribution Distribution(0.0, 1.0);

		for (size_t ArrayIndex = 0; ArrayIndex < ArraySize - 1; ArrayIndex += 2)
		{
			const double Element0 = StandardDeviation * std::sqrt(-2 * std::log(Distribution(RandomEngine)));
			const double Element1 = TWO_PI * Distribution(RandomEngine);

			Array[ArrayIndex] = Element0 * std::cos(Element1) + Mean;
			Array[ArrayIndex + 1] = Element0 * std::sin(Element1) + Mean; 
		}

		// Randomise last element of an uneven array
		if (ArraySize % 2 == 1)
		{
			const double Element0 = StandardDeviation * std::sqrt(-2 * std::log(Distribution(RandomEngine)));
			const double Element1 = TWO_PI * Distribution(RandomEngine);

			Array[ArraySize - 1] = Element0 * std::cos(Element1) + Mean;
		}
	}
}
