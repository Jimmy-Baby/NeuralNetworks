#include "XorMat.h"
#include "../Math.h"
#include "../SimpleLayer.h"

#include <cstdio>
#include <format>
#include <vector>

namespace XorMat
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

#define TRAINING_DATA g_TrainingDataXor

	Matrix g_TrainingData = TRAINING_DATA;

	class XorModel
	{
	public:
		XorModel(const bool Randomise = true)
		{
			HiddenLayers.emplace_back(2, 2, Randomise);
			OutputLayer = Layer(HiddenLayers.back(), 1, Randomise);
		}

		float ForwardToModel(const Matrix& Inputs)
		{
			if (HiddenLayers.empty())
			{
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

			return OutputLayer.GetActivations().At(0, 0);
		}

		void Print() const
		{
			for (auto& Layer : HiddenLayers)
			{
				Layer.PrintValues();
			}

			OutputLayer.PrintValues();
		}

		float CalculateCost()
		{
			float Result = 0.0f;

			for (size_t Index = 0; Index < g_TrainingData.GetRowCount(); ++Index)
			{
				// Get the current training sample
				const float X1 = g_TrainingData.At(Index, 0);
				const float X2 = g_TrainingData.At(Index, 1);
				Matrix TrainingSample({ { X1, X2 } });

				// Use the model to evaluate an output
				const float ModelPrediction = ForwardToModel(TrainingSample);

				// Subtract the correct output from the model's predicted output to get the amount it was 'off' by
				const float PredictionError = ModelPrediction - g_TrainingData.At(Index, 2);

				// Square the error to get a positive number, and add it to the result
				Result += PredictionError * PredictionError;
			}

			// Average out all the accumulated error
			Result /= g_TrainingData.GetRowCount();

			return Result;
		}

		void Learn(const XorModel& Wiggles, const float Rate = 1e-1f)
		{
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

					float& CurrentBias = CurrentLayer.GetBiases().At(0, RowIndex);
					const float CurrentWiggleBias = CurrentWiggleLayer.GetBiases().At(0, RowIndex);
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

		void TrainFiniteDiff(XorModel& Wiggles, const float Epsilon = 1e-3f)
		{
			// Calculate cost
			const float Cost = CalculateCost();

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
						CurrentWiggleLayer.GetWeights().At(RowIndex, ColumnIndex) = (CalculateCost() - Cost) / Epsilon;
						CurrentWeight = Saved;
					}
				}

				for (size_t ColumnIndex = 0; ColumnIndex < CurrentLayer.GetBiases().GetRowCount(); ++ColumnIndex)
				{
					float& CurrentBias = CurrentLayer.GetBiases().At(0, ColumnIndex);
					Saved = CurrentBias;
					CurrentBias += Epsilon;
					CurrentWiggleLayer.GetBiases().At(0, ColumnIndex) = (CalculateCost() - Cost) / Epsilon;
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
					Wiggles.OutputLayer.GetWeights().At(RowIndex, ColumnIndex) = (CalculateCost() - Cost) / Epsilon;
					CurrentWeight = Saved;
				}
			}

			for (size_t ColumnIndex = 0; ColumnIndex < OutputLayer.GetBiases().GetColumnCount(); ++ColumnIndex)
			{
				float& CurrentBias = OutputLayer.GetBiases().At(0, ColumnIndex);
				Saved = CurrentBias;
				CurrentBias += Epsilon;
				Wiggles.OutputLayer.GetBiases().At(0, ColumnIndex) = (CalculateCost() - Cost) / Epsilon;
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
		XorModel Model;

		printf("Starting Cost = %f\n", Model.CalculateCost());
		printf("------------------------------\n");

		// Finite Difference
		for (size_t Index = 0; Index < 1; ++Index)
		{
			// Train
			XorModel Wiggles(false);
			Model.TrainFiniteDiff(Wiggles, 1e-3f);
			Model.Learn(Wiggles, 1.0f);

			// Calculate the new cost after adjusting
			const float NewCost = Model.CalculateCost();

			// Print results of this iteration
			//if ((Index + 1) % 1000 == 0)
			//{
				printf("%zu: Cost = %f\n", Index + 1, NewCost);
			//}

			if (std::format("{:.6f}", NewCost) == "0.000000")
			{
				printf("Breaking!\n");
				break;
			}
		}

		printf("------------------------------\n");
		printf("Model:\n");
		Model.Print();
		printf("------------------------------\n");

		printf("Results:\n");

		for (int Index = 0; Index < 4; ++Index)
		{
			const float X1 = g_TrainingData.At(Index, 0);
			const float X2 = g_TrainingData.At(Index, 1);
			const float Z1 = Model.ForwardToModel(Matrix({ { X1, X2 } }));

			printf("{ %f ^ %f = %f }\n", X1, X2, Z1);
		}
	}
}
