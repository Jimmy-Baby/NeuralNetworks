#pragma once

#include "Types.h"

class Wiggler
{
public:
	virtual ~Wiggler()
	{
	}

	virtual void Reset()
	{
	}

	/// \param GradientVector The gradient for the parameters.
	/// \param ParameterVector Vector of parameters, input and output.
	virtual void UpdateParameterVector(TVector::ConstAlignedMapType& GradientVector, TVector::AlignedMapType& ParameterVector) = 0;
};
