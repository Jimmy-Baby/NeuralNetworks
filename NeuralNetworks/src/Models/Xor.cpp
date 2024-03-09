#include "LogicGates.h"
#include "../Math.h"

#include <cstdio>
#include <cstdlib>
#include <format>

namespace Xor
{
	using TSample = float[3];

	TSample g_TrainingDataOr[] =
	{
		{ 0, 0, 0 },
		{ 1, 0, 1 },
		{ 0, 1, 1 },
		{ 1, 1, 1 },
	};

	TSample g_TrainingDataAnd[] =
	{
		{ 0, 0, 0 },
		{ 1, 0, 0 },
		{ 0, 1, 0 },
		{ 1, 1, 1 },
	};

	TSample g_TrainingDataNand[] =
	{
		{ 0, 0, 1 },
		{ 1, 0, 1 },
		{ 0, 1, 1 },
		{ 1, 1, 0 },
	};

	TSample g_TrainingDataXor[] =
	{
		{ 0, 0, 0 },
		{ 1, 0, 1 },
		{ 0, 1, 1 },
		{ 1, 1, 0 },
	};

#define TRAINING_DATA g_TrainingDataXor

	TSample* g_TrainingData = TRAINING_DATA;
	constexpr size_t TRAINING_DATA_SIZE = std::size(TRAINING_DATA);

	class XorModel
	{
	public:
		XorModel()
		/*: OrWeight1(Math::RandomFloat()),
		  OrWeight2(Math::RandomFloat()),
		  OrBias(Math::RandomFloat()),
		  NandWeight1(Math::RandomFloat()),
		  NandWeight2(Math::RandomFloat()),
		  NandBias(Math::RandomFloat()),
		  AndWeight1(Math::RandomFloat()),
		  AndWeight2(Math::RandomFloat()),
		  AndBias(Math::RandomFloat())*/
			: OrWeight1(0.5f),
			  OrWeight2(0.5f),
			  OrBias(0.5f),
			  NandWeight1(0.5f),
			  NandWeight2(0.5f),
			  NandBias(0.5f),
			  AndWeight1(0.5f),
			  AndWeight2(0.5f),
			  AndBias(0.5f)
		{
		}

		float ForwardToModel(const float X1, const float X2) const
		{
			// Use the model to evaluate an output
			const float A1 = Math::Sigmoid(X1 * OrWeight1 + X2 * OrWeight2 + OrBias);
			const float A2 = Math::Sigmoid(X1 * NandWeight1 + X2 * NandWeight2 + NandBias);
			const float ModelPrediction = Math::Sigmoid(A1 * AndWeight1 + A2 * AndWeight2 + AndBias);

			return ModelPrediction;
		}

		void PrintWeightsAndBiases() const
		{
			printf("Or:   %f | %f | %f\n", OrWeight1, OrWeight2, OrBias);
			printf("Nand: %f | %f | %f\n", NandWeight1, NandWeight2, NandBias);
			printf("And:  %f | %f | %f\n", AndWeight1, AndWeight2, AndBias);
		}

		void PrintIndividualResults() const
		{
			printf("Or:\n");
			for (int Index = 0; Index < 4; ++Index)
			{
				const float X1 = g_TrainingDataOr[Index][0];
				const float X2 = g_TrainingDataOr[Index][1];

				printf("{ %f | %f = %f }\n", X1, X2, X1 * OrWeight1 + X2 * OrWeight2 + OrBias);
			}

			printf("------------------------------\n");

			printf("Nand:\n");
			for (int Index = 0; Index < 4; ++Index)
			{
				const float X1 = g_TrainingDataNand[Index][0];
				const float X2 = g_TrainingDataNand[Index][1];

				printf("{ %f | %f = %f }\n", X1, X2, X1 * NandWeight1 + X2 * NandWeight2 + NandBias);
			}

			printf("------------------------------\n");

			printf("And:\n");
			for (int Index = 0; Index < 4; ++Index)
			{
				const float X1 = g_TrainingDataAnd[Index][0];
				const float X2 = g_TrainingDataAnd[Index][1];

				printf("{ %f | %f = %f }\n", X1, X2, X1 * AndWeight1 + X2 * AndWeight2 + AndBias);
			}
		}

		float CalculateCost() const
		{
			float Result = 0.0f;

			for (size_t Index = 0; Index < TRAINING_DATA_SIZE; ++Index)
			{
				// Get the current training sample
				const float X1 = g_TrainingData[Index][0];
				const float X2 = g_TrainingData[Index][1];

				// Use the model to evaluate an output
				const float A1 = Math::Sigmoid(X1 * OrWeight1 + X2 * OrWeight2 + OrBias);
				const float A2 = Math::Sigmoid(X1 * NandWeight1 + X2 * NandWeight2 + NandBias);
				const float ModelPrediction = Math::Sigmoid(A1 * AndWeight1 + A2 * AndWeight2 + AndBias);

				// Subtract the correct output from the model's predicted output to get the amount it was 'off' by
				const float PredictionError = ModelPrediction - g_TrainingData[Index][2];

				// Square the error to get a positive number, and add it to the result
				Result += PredictionError * PredictionError;
			}

			// Average out all the accumulated error
			Result /= TRAINING_DATA_SIZE;

			return Result;
		}

		void TrainFiniteDiff(const float Rate = 1e-3f, const float Epsilon = 1e-3f)
		{
			// Calculate cost
			const float Cost = CalculateCost();

			// Finite Difference -> f(a + h) - f(a)
			// Calculate The Wiggles(R)
			float Saved = OrWeight1;
			OrWeight1 += Epsilon;
			const float OrWiggleWeight1 = (CalculateCost() - Cost) / Epsilon;
			OrWeight1 = Saved;

			Saved = OrWeight2;
			OrWeight2 += Epsilon;
			const float OrWiggleWeight2 = (CalculateCost() - Cost) / Epsilon;
			OrWeight2 = Saved;

			Saved = OrBias;
			OrBias += Epsilon;
			const float OrWiggleBias = (CalculateCost() - Cost) / Epsilon;
			OrBias = Saved;

			Saved = NandWeight1;
			NandWeight1 += Epsilon;
			const float NandWiggleWeight1 = (CalculateCost() - Cost) / Epsilon;
			NandWeight1 = Saved;

			Saved = NandWeight2;
			NandWeight2 += Epsilon;
			const float NandWiggleWeight2 = (CalculateCost() - Cost) / Epsilon;
			NandWeight2 = Saved;

			Saved = NandBias;
			NandBias += Epsilon;
			const float NandWiggleBias = (CalculateCost() - Cost) / Epsilon;
			NandBias = Saved;

			Saved = AndWeight1;
			AndWeight1 += Epsilon;
			const float AndWiggleWeight1 = (CalculateCost() - Cost) / Epsilon;
			AndWeight1 = Saved;

			Saved = AndWeight2;
			AndWeight2 += Epsilon;
			const float AndWiggleWeight2 = (CalculateCost() - Cost) / Epsilon;
			AndWeight2 = Saved;

			Saved = AndBias;
			AndBias += Epsilon;
			const float AndWiggleBias = (CalculateCost() - Cost) / Epsilon;
			AndBias = Saved;

			// Adjust the weights and bias using The Wiggles(R)
			OrWeight1 -= Rate * OrWiggleWeight1;
			OrWeight2 -= Rate * OrWiggleWeight2;
			OrBias -= Rate * OrWiggleBias;
			NandWeight1 -= Rate * NandWiggleWeight1;
			NandWeight2 -= Rate * NandWiggleWeight2;
			NandBias -= Rate * NandWiggleBias;
			AndWeight1 -= Rate * AndWiggleWeight1;
			AndWeight2 -= Rate * AndWiggleWeight2;
			AndBias -= Rate * AndWiggleBias;
		}

	private:
		float OrWeight1;
		float OrWeight2;
		float OrBias;
		float NandWeight1;
		float NandWeight2;
		float NandBias;
		float AndWeight1;
		float AndWeight2;
		float AndBias;
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

		printf("Cost = %f\n", Model.CalculateCost());
		printf("------------------------------\n");
		
		// Finite Difference
		for (size_t Index = 0; Index < 1000; ++Index)
		{
			// Train
			Model.TrainFiniteDiff(1.0f, 1e-3f);

			// Calculate the new cost after adjusting
			const float NewCost = Model.CalculateCost();

			// Print results of this iteration
			printf("Cost = %f\n", NewCost);

			if (std::format("{:.6f}", NewCost) == "0.000000")
			{
				printf("Breaking!\n");
				break;
			}
		}

		printf("------------------------------\n");
		Model.PrintWeightsAndBiases();
		printf("------------------------------\n");

		printf("Results:\n");

		for (int Index = 0; Index < 4; ++Index)
		{
			const float X1 = g_TrainingData[Index][0];
			const float X2 = g_TrainingData[Index][1];
			const float Z1 = Model.ForwardToModel(X1, X2);

			printf("{ %f ^ %f = %f }\n", X1, X2, Z1);
		}
	}
}
