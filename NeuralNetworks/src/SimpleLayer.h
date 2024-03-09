#pragma once

#include "Matrix.h"

#include <vector>

class Layer
{
public:
	Layer();
	Layer(const Matrix& PreviousLayer, size_t NeuronCount, bool Randomise = false);
	Layer(const Layer& PreviousLayer, size_t NeuronCount, bool Randomise = false);
	Layer(size_t PreviousLayerNeuronCount, size_t NeuronCount, bool Randomise = false);
	Layer(const Layer& Source);
	Layer& operator=(const Layer& Rhs);
	Layer& operator=(Layer&& Rhs) noexcept;
	~Layer();

public:
	void Activate(const Matrix& Inputs);
	void Activate(const Layer& Inputs);
	const Matrix& GetWeights() const;
	const Matrix& GetBiases() const;
	const Matrix& GetActivations() const;
	Matrix& GetWeights();
	Matrix& GetBiases();
	Matrix& GetActivations();
	void PrintValues(int32_t LayerId = -1, const char* FormatSpecifier = "%f") const;

private:
	size_t LayerIdentifier;
	Matrix Weights;
	Matrix Biases;
	Matrix Activations;
};

#define DEFINE_MATRIX(Name, nRowCount, nColumnCount, bRandomise) Matrix Name = Matrix(#Name, nRowCount, nColumnCount, bRandomise)
