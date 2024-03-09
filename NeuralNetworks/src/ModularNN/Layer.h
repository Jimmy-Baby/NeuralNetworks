#pragma once

#include "Types.h"
#include "Wiggler.h"

class Layer
{
public:
	Layer(size_t InputsCount, size_t OutputsCount);
	virtual ~Layer();

	// Getters 
	size_t InputsCount() const;
	size_t OutputsCount() const;

	virtual void Initialise() = 0;
	virtual void Initialise(const TScalar& Mean, const TScalar& StandardDeviation) = 0;
	virtual void Forward(const TMatrix& PreviousLayerMatrix) = 0;
	virtual const TMatrix& Output() const = 0;
	virtual void Backprop(const TMatrix& PreviousLayerMatrix, const TMatrix& NextLayerMatrix) = 0;
	virtual const TMatrix& BackpropData() const = 0;
	virtual void Update(Wiggler& WigglerObject) = 0;
	virtual std::vector<TScalar> GetParameters() const = 0;

private:
	size_t InputsCount_;
	size_t OutputsCount_;
};
