#pragma once

#include "Types.h"

namespace Random
{
	// Fill array with random numbers using Box-Muller transform
	void RandomScalarFill(TScalar* Array, size_t ArraySize, const TScalar& Mean = 0, const TScalar& StandardDeviation = 1);
}
