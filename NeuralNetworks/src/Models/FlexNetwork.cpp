#include "FlexNetwork.h"
#include "../Math.h"
#include "../SimpleLayer.h"

#include <bitset>
#include <cassert>
#include <cstdio>
#include <format>
#include <vector>

namespace FlexNetwork
{
	Matrix g_TrainingDataOr
	({
		{ 0, 0, 0 },
		{ 1, 0, 1 },
		{ 0, 1, 1 },
		{ 1, 1, 1 },
	});

	Matrix g_TrainingDataAnd
	({
		{ 0, 0, 0 },
		{ 1, 0, 0 },
		{ 0, 1, 0 },
		{ 1, 1, 1 },
	});

	Matrix g_TrainingDataNand
	({
		{ 0, 0, 1 },
		{ 1, 0, 1 },
		{ 0, 1, 1 },
		{ 1, 1, 0 },
	});

	Matrix g_TrainingDataXor
	({
		{ 0, 0, 0 },
		{ 1, 0, 1 },
		{ 0, 1, 1 },
		{ 1, 1, 0 },
	});

	Matrix g_TrainingDataAdder
	({
		{ 0, 0, 0, 0, 0 },
		{ 1, 0, 0, 1, 0 },
		{ 0, 1, 0, 1, 0 },
		{ 0, 0, 1, 1, 0 },
		{ 1, 1, 0, 0, 1 },
		{ 0, 1, 1, 0, 1 },
		{ 1, 0, 1, 0, 1 },
		{ 1, 1, 1, 1, 1 },
	});

	Matrix g_TrainingDataAdder4
	({
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
		{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
		{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
		{ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
		{ 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0 },
		{ 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 },
		{ 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0 },
		{ 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0 },
		{ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0 },
		{ 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 },
		{ 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 },
		{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 },
		{ 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1 },
		{ 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0 },
		{ 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1 },
		{ 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0 },
		{ 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1 },
		{ 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
		{ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
		{ 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0 },
		{ 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0 },
		{ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
		{ 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0 },
		{ 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0 },
		{ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
		{ 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0 },
		{ 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
		{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0 },
		{ 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
		{ 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0 },
		{ 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0 },
		{ 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0 },
		{ 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0 },
		{ 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0 },
		{ 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0 },
		{ 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0 },
		{ 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0 },
		{ 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0 },
		{ 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0 },
		{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1 },
		{ 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0 },
		{ 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0 },
		{ 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1 },
		{ 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0 },
		{ 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
	});

	struct TFlexNetworkArchitecture
	{
		size_t InputLayerSize;
		std::vector<size_t> HiddenLayerSizes;
		size_t OutputLayerSize;
	};

	class FlexNetwork
	{
	public:
		explicit FlexNetwork(const TFlexNetworkArchitecture& Architecture, const bool Randomise = true)
		{
			if (Architecture.HiddenLayerSizes.empty())
			{
				OutputLayer = Layer(Architecture.InputLayerSize, Architecture.OutputLayerSize, Randomise);
			}
			else
			{
				HiddenLayers.emplace_back(Architecture.InputLayerSize, Architecture.HiddenLayerSizes[0], Randomise);

				for (size_t HiddenLayerIndex = 1; HiddenLayerIndex < Architecture.HiddenLayerSizes.size(); ++HiddenLayerIndex)
				{
					HiddenLayers.emplace_back(HiddenLayers[HiddenLayerIndex - 1], Architecture.HiddenLayerSizes[HiddenLayerIndex], Randomise);
				}

				OutputLayer = Layer(HiddenLayers.back(), Architecture.OutputLayerSize, Randomise);
			}
		}

		~FlexNetwork()
		{
			//for (Layer& LayerToDelete : HiddenLayers)
			//{
			//	LayerToDelete.~Layer();
			//}

			//OutputLayer.~Layer();
		}

		bool operator==(const FlexNetwork& Rhs) const
		{
			if (HiddenLayers.size() != Rhs.HiddenLayers.size())
			{
				return false;
			}

			for (size_t HiddenLayerIndex = 0; HiddenLayerIndex < HiddenLayers.size(); ++HiddenLayerIndex)
			{
				if (HiddenLayers[HiddenLayerIndex].GetWeights().GetColumnCount() != Rhs.HiddenLayers[HiddenLayerIndex].GetWeights().GetColumnCount())
				{
					return false;
				}

				if (HiddenLayers[HiddenLayerIndex].GetWeights().GetRowCount() != Rhs.HiddenLayers[HiddenLayerIndex].GetWeights().GetRowCount())
				{
					return false;
				}
			}

			if (OutputLayer.GetWeights().GetColumnCount() != Rhs.OutputLayer.GetWeights().GetColumnCount())
			{
				return false;
			}

			if (OutputLayer.GetWeights().GetRowCount() != Rhs.OutputLayer.GetWeights().GetRowCount())
			{
				return false;
			}

			return true;
		}

		bool operator!=(const FlexNetwork& Rhs) const
		{
			return !operator==(Rhs);
		}

		void ForwardToModel(const Matrix& Inputs)
		{
			if (HiddenLayers.empty())
			{
				OutputLayer.Activate(Inputs);
			}
			else
			{
				HiddenLayers[0].Activate(Inputs);

				for (size_t HiddenLayerIndex = 1; HiddenLayerIndex < HiddenLayers.size(); ++HiddenLayerIndex)
				{
					HiddenLayers[HiddenLayerIndex].Activate(HiddenLayers[HiddenLayerIndex - 1]);
				}

				OutputLayer.Activate(HiddenLayers.back());
			}
		}

		const Matrix& GetOutputs() const
		{
			return OutputLayer.GetActivations();
		}

		void Print(const bool PrintHiddenLayers = true, const bool PrintOutputLayer = true) const
		{
			size_t LayerIndex = 0;

			if (PrintHiddenLayers)
			{
				for (; LayerIndex < HiddenLayers.size(); ++LayerIndex)
				{
					HiddenLayers[LayerIndex].PrintValues(static_cast<int32_t>(LayerIndex));
					printf("\n");
				}
			}

			if (PrintOutputLayer)
			{
				OutputLayer.PrintValues(static_cast<int32_t>(LayerIndex));
			}
		}

		float CalculateCost(const Matrix& Inputs, const Matrix& ExpectedOutputs)
		{
			float Result = 0.0f;

			for (size_t InputIndex = 0; InputIndex < Inputs.GetRowCount(); ++InputIndex)
			{
				// Get the current training sample
				Matrix TrainingSample = Inputs.SubMatrix(InputIndex, 0, 1, Inputs.GetColumnCount());

				// Use the model to evaluate an output
				ForwardToModel(TrainingSample);
				const Matrix ModelPredictions = OutputLayer.GetActivations();

				for (size_t OutputIndex = 0; OutputIndex < ModelPredictions.GetColumnCount(); ++OutputIndex)
				{
					// Subtract the correct output from the model's predicted output to get the amount it was 'off' by
					const float ModelPrediction = ModelPredictions.At(0, OutputIndex);
					const float ExpectedOutput = ExpectedOutputs.At(InputIndex, OutputIndex);
					const float PredictionError = ModelPrediction - ExpectedOutput;

					// Square the error to get a positive number, and add it to the result
					Result += PredictionError * PredictionError;
				}
			}

			// Average out all the accumulated error
			Result /= Inputs.GetRowCount();

			return Result;
		}

		void Learn(const FlexNetwork& Wiggles, const float Rate = 1e-1f)
		{
			if (*this != Wiggles)
			{
				throw std::format_error("Wiggles network does not match layout");
			}

			for (size_t LayerIndex = 0; LayerIndex < HiddenLayers.size(); ++LayerIndex)
			{
				Layer& CurrentLayer = HiddenLayers[LayerIndex];
				const Layer& CurrentWiggleLayer = Wiggles.HiddenLayers[LayerIndex];

				for (size_t RowIndex = 0; RowIndex < CurrentLayer.GetWeights().GetRowCount(); ++RowIndex)
				{
					for (size_t ColumnIndex = 0; ColumnIndex < CurrentLayer.GetWeights().GetColumnCount(); ++ColumnIndex)
					{
						float& CurrentWeight = CurrentLayer.GetWeights().At(RowIndex, ColumnIndex);
						const float CurrentWiggleWeight = CurrentWiggleLayer.GetWeights().At(RowIndex, ColumnIndex);
						CurrentWeight -= Rate * CurrentWiggleWeight;
					}
				}

				for (size_t ColumnIndex = 0; ColumnIndex < CurrentLayer.GetBiases().GetColumnCount(); ++ColumnIndex)
				{
					float& CurrentBias = CurrentLayer.GetBiases().At(0, ColumnIndex);
					const float CurrentWiggleBias = CurrentWiggleLayer.GetBiases().At(0, ColumnIndex);
					CurrentBias -= Rate * CurrentWiggleBias;
				}
			}

			for (size_t RowIndex = 0; RowIndex < OutputLayer.GetWeights().GetRowCount(); ++RowIndex)
			{
				for (size_t ColumnIndex = 0; ColumnIndex < OutputLayer.GetWeights().GetColumnCount(); ++ColumnIndex)
				{
					float& CurrentWeight = OutputLayer.GetWeights().At(RowIndex, ColumnIndex);
					const float CurrentWiggleWeight = Wiggles.OutputLayer.GetWeights().At(RowIndex, ColumnIndex);
					CurrentWeight -= Rate * CurrentWiggleWeight;
				}
			}

			for (size_t ColumnIndex = 0; ColumnIndex < OutputLayer.GetBiases().GetColumnCount(); ++ColumnIndex)
			{
				float& CurrentBias = OutputLayer.GetBiases().At(0, ColumnIndex);
				const float CurrentWiggleBias = Wiggles.OutputLayer.GetBiases().At(0, ColumnIndex);
				CurrentBias -= Rate * CurrentWiggleBias;
			}
		}

		void TrainFiniteDiff(FlexNetwork& Wiggles, const Matrix& TrainingInputs, const Matrix& TrainingOutputs, const float Epsilon = 1e-3f)
		{
			if (*this != Wiggles)
			{
				throw std::format_error("Wiggles network does not match layout");
			}

			// Calculate cost
			const float Cost = CalculateCost(TrainingInputs, TrainingOutputs);

			float Saved;

			// Finite Difference -> f(a + h) - f(a)

			// Process hidden layer training
			for (size_t LayerIndex = 0; LayerIndex < HiddenLayers.size(); ++LayerIndex)
			{
				Layer& CurrentLayer = HiddenLayers[LayerIndex];
				Layer& CurrentWiggleLayer = Wiggles.HiddenLayers[LayerIndex];

				for (size_t RowIndex = 0; RowIndex < CurrentLayer.GetWeights().GetRowCount(); ++RowIndex)
				{
					for (size_t ColumnIndex = 0; ColumnIndex < CurrentLayer.GetWeights().GetColumnCount(); ++ColumnIndex)
					{
						float& CurrentWeight = CurrentLayer.GetWeights().At(RowIndex, ColumnIndex);
						Saved = CurrentWeight;
						CurrentWeight += Epsilon;
						CurrentWiggleLayer.GetWeights().At(RowIndex, ColumnIndex) = (CalculateCost(TrainingInputs, TrainingOutputs) - Cost) / Epsilon;
						CurrentWeight = Saved;
					}
				}

				for (size_t ColumnIndex = 0; ColumnIndex < CurrentLayer.GetBiases().GetColumnCount(); ++ColumnIndex)
				{
					float& CurrentBias = CurrentLayer.GetBiases().At(0, ColumnIndex);
					Saved = CurrentBias;
					CurrentBias += Epsilon;
					CurrentWiggleLayer.GetBiases().At(0, ColumnIndex) = (CalculateCost(TrainingInputs, TrainingOutputs) - Cost) / Epsilon;
					CurrentBias = Saved;
				}
			}

			// Process output layer training
			for (size_t RowIndex = 0; RowIndex < OutputLayer.GetWeights().GetRowCount(); ++RowIndex)
			{
				for (size_t ColumnIndex = 0; ColumnIndex < OutputLayer.GetWeights().GetColumnCount(); ++ColumnIndex)
				{
					float& CurrentWeight = OutputLayer.GetWeights().At(RowIndex, ColumnIndex);
					Saved = CurrentWeight;
					CurrentWeight += Epsilon;
					Wiggles.OutputLayer.GetWeights().At(RowIndex, ColumnIndex) = (CalculateCost(TrainingInputs, TrainingOutputs) - Cost) / Epsilon;
					CurrentWeight = Saved;
				}
			}

			for (size_t ColumnIndex = 0; ColumnIndex < OutputLayer.GetBiases().GetColumnCount(); ++ColumnIndex)
			{
				float& CurrentBias = OutputLayer.GetBiases().At(0, ColumnIndex);
				Saved = CurrentBias;
				CurrentBias += Epsilon;
				Wiggles.OutputLayer.GetBiases().At(0, ColumnIndex) = (CalculateCost(TrainingInputs, TrainingOutputs) - Cost) / Epsilon;
				CurrentBias = Saved;
			}
		}

	private:
		std::vector<Layer> HiddenLayers;
		Layer OutputLayer;
	};

	// Expected formula:
	// y = x1 * 0.5 + 
	//
	// Model formula:
	// y = x * w

	void Run()
	{
		// Create our model
		const TFlexNetworkArchitecture Architecture = { 3, { 5, 4 }, 2 };
		FlexNetwork Model(Architecture);

		// Training
		{
			printf("Pre-trained Model:\n\n");
			Model.Print();

			printf("\nTraining Inputs:\n\n");
			const Matrix TrainingInputs = g_TrainingDataAdder.SubMatrix(0, 0, 8, 3);
			TrainingInputs.PrintValues();

			printf("\nTraining Outputs:\n\n");
			const Matrix TrainingOutputs = g_TrainingDataAdder.SubMatrix(0, 3, 8, 2);
			TrainingOutputs.PrintValues();

			printf("\nStarting Cost = %f\n", Model.CalculateCost(TrainingInputs, TrainingOutputs));
			printf("------------------------------\n");

			// Finite Difference
			for (size_t Index = 0; Index < 10000; ++Index)
			{
				// Train
				FlexNetwork Wiggles(Architecture, false);
				Model.TrainFiniteDiff(Wiggles, TrainingInputs, TrainingOutputs, 1e-3f);
				Model.Learn(Wiggles, 1.0f);

				// Calculate the new cost after adjusting
				const float NewCost = Model.CalculateCost(TrainingInputs, TrainingOutputs);

				// Print results of this iteration
				//if (Index + 1 % 1000 == 0)
				//{
				printf("%zu: Cost = %f\n", Index + 1, NewCost);
				//}

				if (std::format("{:.6f}", NewCost) == "0.000000")
				{
					printf("Breaking!\n");
					break;
				}
			}
		}

		printf("------------------------------\n");

		printf("Post-trained Model:\n\n");
		Model.Print();

		printf("------------------------------\n");

		// Testing
		{
			Matrix TestingInputs = g_TrainingDataAdder.SubMatrix(0, 0, 8, 3);

			printf("Test:\n");
			for (size_t LhsIndex = 0; LhsIndex < 2; ++LhsIndex)
			{
				for (size_t RhsIndex = 0; RhsIndex < 2; ++RhsIndex)
				{
					// Matrix to store inputs
					Matrix ForwardMatrix(1, 3);

					// Create inputs for 
					std::bitset<8> LhsBitset(LhsIndex);
					std::bitset<8> RhsBitset(RhsIndex);

					ForwardMatrix.At(0, 1) = LhsBitset.test(0);
					//ForwardMatrix.At(0, 2) = LhsBitset.test(1);
					//ForwardMatrix.At(0, 3) = LhsBitset.test(0) && LhsBitset.test(1);

					ForwardMatrix.At(0, 2) = RhsBitset.test(0);
					//ForwardMatrix.At(0, 6) = RhsBitset.test(1);

					// Feed data to model
					Model.ForwardToModel(ForwardMatrix);
					const Matrix& Outputs = Model.GetOutputs();

					std::bitset<8> Result;
					Result.set(0, Outputs.At(0, 0) > 0.95f);
					Result.set(1, Outputs.At(0, 1) > 0.95f);

					if (Outputs.At(0, 4) > 0.95f)
					{
						printf("{ %llu + %llu = %llu | OVERFLOW }\n", LhsIndex, RhsIndex, Result.to_ullong());
					}
					else
					{
						printf("{ %llu + %llu = %llu | NOOVERFLOW }\n", LhsIndex, RhsIndex, Result.to_ullong());
					}
				}
			}
		}
	}
}
