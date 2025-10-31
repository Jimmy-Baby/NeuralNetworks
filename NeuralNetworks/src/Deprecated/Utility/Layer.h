#pragma once

#include "Matrix.h"

class Layer
{
public:
	Layer();
	Layer(const Matrix& previousLayer, size_t neuronCount, bool randomise = false);
	Layer(const Layer& previousLayer, size_t neuronCount, bool randomise = false);
	Layer(size_t previousLayerNeuronCount, size_t neuronCount, bool randomise = false);
	Layer(const Layer& source);
	Layer(Layer&& source) noexcept;
	Layer& operator=(const Layer& rhs);
	Layer& operator=(Layer&& rhs) noexcept;

public:
	void Activate(const Matrix& inputs);
	void Activate(const Layer& inputs);
	[[nodiscard]] const Matrix& GetWeights() const;
	[[nodiscard]] const Matrix& GetBiases() const;
	[[nodiscard]] const Matrix& GetActivations() const;
	[[nodiscard]] Matrix& GetWeights();
	[[nodiscard]] Matrix& GetBiases();
	[[nodiscard]] Matrix& GetActivations();
	void PrintValues(int32_t layerId = -1, const char* formatSpecifier = "%f") const;

private:
	size_t LayerIdentifier;
	Matrix Weights;
	Matrix Biases;
	Matrix Activations;

	static size_t LayerIdCounter;
};

#define DEFINE_MATRIX(Name, nRowCount, nColumnCount, bRandomise) Matrix Name = Matrix(#Name, nRowCount, nColumnCount, bRandomise)
