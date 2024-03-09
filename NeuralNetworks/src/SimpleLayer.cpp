#include "SimpleLayer.h"
#include "Matrix.h"
#include "Math.h"

#include <format>

static size_t g_LayerIdentifierCounter = 0;

Layer::Layer()
	: Layer(0, 0, false)
{
}

Layer::Layer(const Matrix& PreviousLayer, const size_t NeuronCount, const bool Randomise)
	: Layer(PreviousLayer.GetColumnCount(), NeuronCount, Randomise)
{
}

Layer::Layer(const Layer& PreviousLayer, const size_t NeuronCount, const bool Randomise)
	: Layer(PreviousLayer.Activations.GetColumnCount(), NeuronCount, Randomise)
{
}

Layer::Layer(const size_t PreviousLayerNeuronCount, const size_t NeuronCount, const bool Randomise)
	: LayerIdentifier(g_LayerIdentifierCounter++),
	  Weights(PreviousLayerNeuronCount, NeuronCount, Randomise),
	  Biases(1, NeuronCount, Randomise),
	  Activations(1, NeuronCount, Randomise)
{
}

Layer::Layer(const Layer& Source)
{
	LayerIdentifier = g_LayerIdentifierCounter++;
	this->Weights = Source.Weights;
	this->Biases = Source.Biases;
	this->Activations = Source.Activations;
}

Layer& Layer::operator=(const Layer& Rhs)
{
	this->Weights = Rhs.Weights;
	this->Biases = Rhs.Biases;
	this->Activations = Rhs.Activations;

	return *this;
}

Layer& Layer::operator=(Layer&& Rhs) noexcept
{
	LayerIdentifier = g_LayerIdentifierCounter++;
	Weights = std::move(Rhs.Weights);
	Biases = std::move(Rhs.Biases);
	Activations = std::move(Rhs.Activations);

	return *this;
}

Layer::~Layer()
{
	//Weights.~Matrix();
	//Biases.~Matrix();
	//Activations.~Matrix();
}


void Layer::Activate(const Matrix& Inputs)
{
	Activations = Inputs * Weights + Biases;
	Activations.Activate();
}

void Layer::Activate(const Layer& Inputs)
{
	Activations = Inputs.Activations * Weights + Biases;
	Activations.Activate();
}

const Matrix& Layer::GetWeights() const
{
	return Weights;
}

const Matrix& Layer::GetBiases() const
{
	return Biases;
}

const Matrix& Layer::GetActivations() const
{
	return Activations;
}

Matrix& Layer::GetWeights()
{
	return Weights;
}

Matrix& Layer::GetBiases()
{
	return Biases;
}

Matrix& Layer::GetActivations()
{
	return Activations;
}

void Layer::PrintValues(const int32_t LayerId, const char* FormatSpecifier) const
{
	if (LayerId == -1)
	{
		Weights.PrintValues("Weights", FormatSpecifier);
		printf("\n");

		Biases.PrintValues("Biases", FormatSpecifier);
		printf("\n");

		Activations.PrintValues("Activations", FormatSpecifier);
	}
	else
	{
		char Buffer[64];

		sprintf_s(Buffer, "Weights[%d]", LayerId);
		Weights.PrintValues(Buffer, FormatSpecifier);
		printf("\n");

		sprintf_s(Buffer, "Biases[%d]", LayerId);
		Biases.PrintValues(Buffer, FormatSpecifier);
		printf("\n");

		sprintf_s(Buffer, "Activations[%d]", LayerId);
		Activations.PrintValues(Buffer, FormatSpecifier);
	}
}
