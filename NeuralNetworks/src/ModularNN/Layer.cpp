#include "Layer.h"

Layer::Layer(const size_t InputsCount, const size_t OutputsCount)
	: InputsCount_(InputsCount), OutputsCount_(OutputsCount)
{
}

Layer::~Layer()
{
}

size_t Layer::InputsCount() const
{
	return InputsCount_;
}

size_t Layer::OutputsCount() const
{
	return OutputsCount_;
}
