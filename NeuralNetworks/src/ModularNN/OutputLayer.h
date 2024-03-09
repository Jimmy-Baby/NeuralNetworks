#pragma once

#include "Layer.h"

class OutputLayer
{
public:
	virtual ~OutputLayer();

	virtual void Evaluate(const TMatrix& PreviousLayerMatrix, const TMatrix& TrainingMatrix) = 0;
	virtual const TMatrix& BackpropData() const = 0;
	virtual TScalar Loss() const = 0;
};
