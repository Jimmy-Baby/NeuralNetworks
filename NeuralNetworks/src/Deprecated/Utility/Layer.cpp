// Precompiled headers
#include "Pch.h"

#include "Layer.h"

size_t Layer::LayerIdCounter = 0;

namespace
{
	// Shared RNG for initialisation
	std::random_device GRandomDevice;
	std::mt19937 GRng(GRandomDevice());

	void FillUniform(const Matrix& matrix, const float a, const float b)
	{
		std::uniform_real_distribution dist(a, b);
		float* data = matrix.GetData();
		const size_t size = matrix.GetRowCount() * matrix.GetColumnCount();
		for (size_t index = 0; index < size; ++index)
		{
			data[index] = dist(GRng);
		}
	}

	void FillNormal(const Matrix& matrix, const float mean, const float stddev)
	{
		std::normal_distribution dist(mean, stddev);
		float* data = matrix.GetData();
		const size_t size = matrix.GetRowCount() * matrix.GetColumnCount();
		for (size_t index = 0; index < size; ++index)
		{
			data[index] = dist(GRng);
		}
	}

	// Xavier Glorot uniform: U(-sqrt(6)/(sqrt(fanIn+fanOut)), +limit)
	void XavierInit(const Matrix& weights, const size_t fanIn, const size_t fanOut)
	{
		const float limit = std::sqrt(6.0f / static_cast<float>(fanIn + fanOut));
		FillUniform(weights, -limit, limit);
	}

	// He normal: N(0, sqrt(2/fanIn))
	void HeInit(const Matrix& weights, const size_t fanIn)
	{
		const float stddev = std::sqrt(2.0f / static_cast<float>(fanIn));
		FillNormal(weights, 0.0f, stddev);
	}
}

Layer::Layer()
	: Layer(0, 0, false) {}

Layer::Layer(const Matrix& previousLayer, const size_t neuronCount, const bool randomise)
	: Layer(previousLayer.GetColumnCount(), neuronCount, randomise) {}

Layer::Layer(const Layer& previousLayer, const size_t neuronCount, const bool randomise)
	: Layer(previousLayer.Activations.GetColumnCount(), neuronCount, randomise) {}

Layer::Layer(const size_t previousLayerNeuronCount, const size_t neuronCount, const bool randomise)
	: LayerIdentifier(LayerIdCounter++),
	  Weights(previousLayerNeuronCount, neuronCount, false),
	  Biases(1, neuronCount, false),
	  Activations(1, neuronCount, false) // Don't randomise activations, they are computed
{
	// Improved initialisation
	if (randomise && previousLayerNeuronCount > 0 && neuronCount > 0)
	{
		// Default activations are sigmoid in this project => Xavier works well
		XavierInit(Weights, previousLayerNeuronCount, neuronCount);
		// Biases left at zero (good default)
	}
}

Layer::Layer(const Layer& source)
	: LayerIdentifier(LayerIdCounter++),
	  Weights(source.Weights),
	  Biases(source.Biases),
	  Activations(source.Activations) {}

Layer::Layer(Layer&& source) noexcept
	: LayerIdentifier(LayerIdCounter++),
	  Weights(std::move(source.Weights)),
	  Biases(std::move(source.Biases)),
	  Activations(std::move(source.Activations)) {}

Layer& Layer::operator=(const Layer& rhs)
{
	if (this == &rhs)
	{
		return *this;
	}

	Weights = rhs.Weights;
	Biases = rhs.Biases;
	Activations = rhs.Activations;

	return *this;
}

Layer& Layer::operator=(Layer&& rhs) noexcept
{
	if (this == &rhs)
	{
		return *this;
	}

	Weights = std::move(rhs.Weights);
	Biases = std::move(rhs.Biases);
	Activations = std::move(rhs.Activations);

	return *this;
}

void Layer::Activate(const Matrix& inputs)
{
	// z = inputs * weights + biases
	// Then apply activation function in-place
	inputs.Dot(Weights, Activations);
	Activations += Biases;
	Activations.Activate();
}

void Layer::Activate(const Layer& inputs)
{
	// z = inputs.activations * weights + biases
	// Then apply activation function in-place
	inputs.Activations.Dot(Weights, Activations);
	Activations += Biases;
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

void Layer::PrintValues(const int32_t layerId, const char* formatSpecifier) const
{
	if (layerId == -1)
	{
		Weights.PrintValues("Weights", formatSpecifier);
		printf("\n");

		Biases.PrintValues("Biases", formatSpecifier);
		printf("\n");

		Activations.PrintValues("Activations", formatSpecifier);
	}
	else
	{
		char buffer[64];

		sprintf_s(buffer, "Weights[%d]", layerId);
		Weights.PrintValues(buffer, formatSpecifier);
		printf("\n");

		sprintf_s(buffer, "Biases[%d]", layerId);
		Biases.PrintValues(buffer, formatSpecifier);
		printf("\n");

		sprintf_s(buffer, "Activations[%d]", layerId);
		Activations.PrintValues(buffer, formatSpecifier);
	}
}
