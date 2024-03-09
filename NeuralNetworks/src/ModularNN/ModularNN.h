#pragma once

#include <vector>
#include <memory>

#include "Types.h"
#include "OutputLayer.h"

class Layer;

class ModularNn
{
public:
	ModularNn();
	~ModularNn();

	// Getters
	size_t LayerCount() const;
	const std::vector<std::unique_ptr<Layer>>& GetLayers() const;

	void Initialise(const TScalar& Mean = 0, const TScalar& StandardDeviation = 0.01);
	void Forward(const TMatrix& Input);
	void Update(Wiggler& WigglerObject);
	void AddLayer(std::unique_ptr<Layer>&& Layer);
	void SetOutput(std::unique_ptr<OutputLayer>&& Output);

	template <typename TargetType>
	void Backprop(const TMatrix& Input, const TargetType& Target)
	{
		const int NumberOfLayers = LayerCount();

		if (NumberOfLayers <= 0)
		{
			return;
		}

		Layer* FirstLayer = HiddenLayers_[0].get();
		Layer* LastLayer = HiddenLayers_[NumberOfLayers - 1].get();

		// Let output layer compute back-propagation data
		Output_->Evaluate(LastLayer->Output(), Target);

		// If there is only one hidden layer, "prev_layer_data" will be the input data
		if (NumberOfLayers == 1)
		{
			FirstLayer->Backprop(Input, Output_->BackpropData());
			return;
		}

		// Compute gradients for the last hidden layer
		LastLayer->Backprop(HiddenLayers_[NumberOfLayers - 2]->Output(), Output_->BackpropData());

		// Compute gradients for all the hidden layers except for the first one and the last one
		for (size_t Index = NumberOfLayers - 2; Index > 0; Index--)
		{
			HiddenLayers_[Index]->Backprop(HiddenLayers_[Index - 1]->Output(), HiddenLayers_[Index + 1]->BackpropData());
		}

		// Compute gradients for the first layer
		FirstLayer->Backprop(Input, HiddenLayers_[1]->BackpropData());
	}

private:
	void VerifyNetwork() const;

	std::vector<std::unique_ptr<Layer>> HiddenLayers_;
	std::unique_ptr<OutputLayer> Output_;
};
