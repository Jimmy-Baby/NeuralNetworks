// Precompiled headers
#include "Pch.h"

#include "TinyDnnDigitRecognizer.h"

namespace TinyDnnDigitRecognizer
{
	using namespace tiny_cnn;

	namespace
	{
		// Constants for network architecture and training
		constexpr cnn_size_t ImageWidth = 28;
		constexpr cnn_size_t ImageHeight = 28;
		constexpr cnn_size_t InputSize = ImageWidth * ImageHeight; // 784 pixels
		constexpr cnn_size_t OutputSize = 10; // 10 digits (0-9)
		
		// CNN architecture parameters
		constexpr cnn_size_t Conv1Filters = 32;  // Number of filters in first conv layer
		constexpr cnn_size_t Conv2Filters = 64;  // Number of filters in second conv layer
		constexpr cnn_size_t Conv1KernelSize = 5; // 5x5 kernel for first conv
		constexpr cnn_size_t Conv2KernelSize = 5; // 5x5 kernel for second conv
		constexpr cnn_size_t PoolSize = 2;        // 2x2 max pooling
		constexpr cnn_size_t FCLayerSize = 128;   // Fully connected layer size

		constexpr int BatchSize = 32;
		constexpr int TrainingEpochs = 10;
		constexpr tiny_cnn::float_t DropoutRate = 0.25;

		/// <summary>
		/// Reads a 32-bit big-endian integer from file stream
		/// </summary>
		uint32_t ReadBigEndianInt32(std::ifstream& file)
		{
			uint32_t value;
			unsigned char bytes[4];
			file.read(reinterpret_cast<char*>(bytes), 4);
			value = (static_cast<uint32_t>(bytes[0]) << 24) | (static_cast<uint32_t>(bytes[1]) << 16) | (static_cast<uint32_t>(bytes[2]) << 8) | static_cast<uint32_t>(bytes[3]);
			return value;
		}

		/// <summary>
		/// Loads MNIST image data from IDX3-UBYTE file format
		/// </summary>
		/// <param name="filename">Path to the IDX3-UBYTE file</param>
		/// <returns>Vector of image vectors, each containing normalized pixel values (0-1)</returns>
		std::vector<vec_t> LoadMnistImages(const std::string& filename)
		{
			std::ifstream file(filename, std::ios::binary);
			if (!file.is_open())
			{
				throw std::runtime_error("Cannot open file: " + filename);
			}

			// Read magic number
			const uint32_t magicNumber = ReadBigEndianInt32(file);
			if (magicNumber != 0x00000803) // 0x00 0x00 0x08 0x03
			{
				throw std::runtime_error("Invalid IDX3-UBYTE magic number in " + filename);
			}

			// Read dimensions
			const uint32_t numImages = ReadBigEndianInt32(file);
			const uint32_t numRows = ReadBigEndianInt32(file);
			const uint32_t numCols = ReadBigEndianInt32(file);

			if (numRows != ImageHeight || numCols != ImageWidth)
			{
				throw std::runtime_error("Image dimensions mismatch. Expected 28x28, got " + std::to_string(numRows) + "x" + std::to_string(numCols));
			}

			std::vector<vec_t> images;
			images.reserve(numImages);

			// Read each image
			std::vector<unsigned char> buffer(InputSize);
			for (uint32_t imageIndex = 0; imageIndex < numImages; ++imageIndex)
			{
				file.read(reinterpret_cast<char*>(buffer.data()), InputSize);

				vec_t image(InputSize);
				for (size_t pixelIndex = 0; pixelIndex < InputSize; ++pixelIndex)
				{
					// Normalise pixel values from [0, 255] to [0, 1]
					image[pixelIndex] = static_cast<tiny_cnn::float_t>(buffer[pixelIndex]) / 255.0;
				}

				images.emplace_back(std::move(image));
			}

			file.close();
			return images;
		}

		/// <summary>
		/// Loads MNIST label data from IDX1-UBYTE file format
		/// </summary>
		/// <param name="filename">Path to the IDX1-UBYTE file</param>
		/// <returns>Vector of one-hot encoded label vectors</returns>
		std::vector<vec_t> LoadMnistLabels(const std::string& filename)
		{
			std::ifstream file(filename, std::ios::binary);
			if (!file.is_open())
			{
				throw std::runtime_error("Cannot open file: " + filename);
			}

			// Read magic number
			const uint32_t magicNumber = ReadBigEndianInt32(file);
			if (magicNumber != 0x00000801) // 0x00 0x00 0x08 0x01
			{
				throw std::runtime_error("Invalid IDX1-UBYTE magic number in " + filename);
			}

			// Read number of labels
			const uint32_t numLabels = ReadBigEndianInt32(file);

			std::vector<vec_t> labels;
			labels.reserve(numLabels);

			// Read each label
			for (uint32_t labelIndex = 0; labelIndex < numLabels; ++labelIndex)
			{
				uint8_t label;
				file.read(reinterpret_cast<char*>(&label), 1);

				if (label >= OutputSize)
				{
					throw std::runtime_error("Invalid label value: " + std::to_string(label));
				}

				// Create one-hot encoded vector
				vec_t oneHot(OutputSize, 0.0);
				oneHot[label] = 1.0;
				labels.emplace_back(std::move(oneHot));
			}

			file.close();
			return labels;
		}

		/// <summary>
		/// Loads MNIST dataset from IDX files
		/// </summary>
		/// <param name="imageFile">Path to images file (IDX3-UBYTE)</param>
		/// <param name="labelFile">Path to labels file (IDX1-UBYTE)</param>
		/// <returns>Pair of image vectors and corresponding label vectors</returns>
		std::pair<std::vector<vec_t>, std::vector<vec_t>> LoadMnistDataset(const std::string& imageFile, const std::string& labelFile)
		{
			std::println("Loading images from: {}", imageFile);
			auto images = LoadMnistImages(imageFile);

			std::println("Loading labels from: {}", labelFile);
			auto labels = LoadMnistLabels(labelFile);

			if (images.size() != labels.size())
			{
				throw std::runtime_error("Number of images (" + std::to_string(images.size()) + ") does not match number of labels (" + std::to_string(labels.size()) + ")");
			}

			std::println("Loaded {} samples", images.size());
			return { images, labels };
		}

		/// <summary>
		/// Converts network output to predicted digit (0-9)
		/// </summary>
		int GetPredictedDigit(const vec_t& output)
		{
			return static_cast<int>(std::distance(output.begin(), std::ranges::max_element(output.begin(), output.end())));
		}

		/// <summary>
		/// Converts one-hot encoded output to digit label
		/// </summary>
		int GetExpectedDigit(const vec_t& output)
		{
			return static_cast<int>(std::distance(output.begin(), std::ranges::max_element(output.begin(), output.end())));
		}

		/// <summary>
		/// Evaluates the trained network's accuracy on the dataset
		/// </summary>
		void EvaluateAndPrintAccuracy(network<sequential>& neuralNetwork, const std::vector<vec_t>& testInputs, const std::vector<vec_t>& expectedOutputs)
		{
			size_t correctPredictions = 0;

			// Confusion matrix for detailed analysis
			std::vector confusionMatrix(OutputSize, std::vector(OutputSize, 0));

			for (size_t sampleIndex = 0; sampleIndex < testInputs.size(); ++sampleIndex)
			{
				const vec_t prediction = neuralNetwork.predict(testInputs[sampleIndex]);

				const int predictedDigit = GetPredictedDigit(prediction);
				const int expectedDigit = GetExpectedDigit(expectedOutputs[sampleIndex]);

				confusionMatrix[expectedDigit][predictedDigit]++;

				if (predictedDigit == expectedDigit)
				{
					++correctPredictions;
				}
			}

			const tiny_cnn::float_t accuracy = 100.0 * static_cast<tiny_cnn::float_t>(correctPredictions) / static_cast<tiny_cnn::float_t>(testInputs.size());

			std::printnl();
			std::println("Results:");
			std::println("  Accuracy: {:.2f}% ({}/{} correct)", accuracy, correctPredictions, testInputs.size());

			std::printnl();
			std::println("Confusion Matrix:");
			std::print("     ");

			for (cnn_size_t outputIndex = 0; outputIndex < OutputSize; ++outputIndex)
			{
				std::print("{:4}", outputIndex);
			}

			std::printnl();

			for (cnn_size_t expectedIndex = 0; expectedIndex < OutputSize; ++expectedIndex)
			{
				std::print("{:4} ", expectedIndex);

				for (cnn_size_t predictedIndex = 0; predictedIndex < OutputSize; ++predictedIndex)
				{
					std::print("{:4}", confusionMatrix[expectedIndex][predictedIndex]);
				}

				std::printnl();
			}
		}

		/// <summary>
		/// Displays a simple ASCII visualisation of the image
		/// </summary>
		void DisplayImage(const vec_t& image)
		{
			const auto shades = " .':-=+*#%@";
			constexpr int numShades = 11;

			for (size_t y = 0; y < ImageHeight; ++y)
			{
				for (size_t x = 0; x < ImageWidth; ++x)
				{
					const size_t pixelIndex = y * ImageWidth + x;
					const tiny_cnn::float_t value = image[pixelIndex];
					const int shadeIndex = static_cast<int>(value * (numShades - 1));
					std::print("{}", shades[std::min(shadeIndex, numShades - 1)]);
				}

				std::printnl();
			}
		}

		/// <summary>
		/// Allow the user to test the network with random samples or custom input
		/// </summary>
		void InteractiveTest(network<sequential>& neuralNetwork, const std::vector<vec_t>& testInputs, const std::vector<vec_t>& testOutputs)
		{
			std::printnl();
			std::println("Interactive test");
			std::println("Commands:");
			std::println("  'r' - Test a random sample from the dataset");
			std::println("  'q' - Quit");

			std::string line;
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution<size_t> dist(0, testInputs.size() - 1);

			for (;;)
			{
				std::print("> ");

				if (!std::getline(std::cin, line))
				{
					break;
				}

				if (line.empty())
				{
					continue;
				}

				if (line == "q" || line == "Q" || line == "quit" || line == "QUIT")
				{
					break;
				}

				if (line == "r" || line == "R")
				{
					// Pick random sample
					const size_t sampleIndex = dist(gen);
					const vec_t& input = testInputs[sampleIndex];
					const vec_t& expectedOutput = testOutputs[sampleIndex];

					std::printnl();
					std::println("Sample #{}", sampleIndex);
					std::println("Image:");
					DisplayImage(input);

					// Predict
					const vec_t prediction = neuralNetwork.predict(input);
					const int predictedDigit = GetPredictedDigit(prediction);
					const int expectedDigit = GetExpectedDigit(expectedOutput);

					std::printnl();
					std::println("Predicted: {}", predictedDigit);
					std::println("Expected:{}", expectedDigit);
					std::println("Match: {}", (predictedDigit == expectedDigit) ? "YES" : "NO");

					std::printnl();
					std::println("Output probabilities:");

					for (cnn_size_t outputIndex = 0; outputIndex < OutputSize; ++outputIndex)
					{
						std::println("  Digit {}: {:.4f}", outputIndex, prediction[outputIndex]);
					}

					std::printnl();
				}
				else
				{
					std::println("Unknown command. Use 'r' for random test or 'q' to quit.");
				}
			}
		}
	}

	void Run()
	{
		std::println("Training Handwritten Digit Recognition Network");
		std::printnl();

		// Define MNIST file paths
		const auto trainImagesFile = "data/train-images.idx3-ubyte";
		const auto trainLabelsFile = "data/train-labels.idx1-ubyte";
		const auto testImagesFile = "data/t10k-images.idx3-ubyte";
		const auto testLabelsFile = "data/t10k-labels.idx1-ubyte";

		std::vector<vec_t> trainingInputs;
		std::vector<vec_t> trainingOutputs;
		std::vector<vec_t> testInputs;
		std::vector<vec_t> testOutputs;

		try
		{
			// Try to load actual MNIST data
			std::println("Loading MNIST training dataset...");
			auto [trainImages, trainLabels] = LoadMnistDataset(trainImagesFile, trainLabelsFile);
			trainingInputs = std::move(trainImages);
			trainingOutputs = std::move(trainLabels);

			std::println("Loading MNIST test dataset...");
			auto [testImages, testLabels] = LoadMnistDataset(testImagesFile, testLabelsFile);
			testInputs = std::move(testImages);
			testOutputs = std::move(testLabels);

			std::printnl();
			std::println("Successfully loaded MNIST dataset:");
			std::println("  Training samples: {}", trainingInputs.size());
		 std::println("  Test samples: {}", testInputs.size());
		}
		catch (const std::exception& e)
		{
			std::printnl();
			std::println("Failed to load MNIST data: {}", e.what());
			std::println("Please ensure MNIST files are in the 'data' directory.");
			return;
		}

		// Build the Convolutional Neural Network
		// Architecture: input -> conv1 -> pool1 -> conv2 -> pool2 -> fc -> dropout -> output
		std::printnl();
		std::println("Building Convolutional Neural Network...");
		std::println("Architecture:");
		std::println("  Input:       28x28x1 (grayscale images)");
		std::println("  Conv1:       {}x{} kernel, {} filters -> 24x24x{}", Conv1KernelSize, Conv1KernelSize, Conv1Filters, Conv1Filters);
		std::println("  MaxPool1:    {}x{} pooling -> 12x12x{}", PoolSize, PoolSize, Conv1Filters);
		std::println("  Conv2:       {}x{} kernel, {} filters -> 8x8x{}", Conv2KernelSize, Conv2KernelSize, Conv2Filters, Conv2Filters);
		std::println("  MaxPool2:    {}x{} pooling -> 4x4x{}", PoolSize, PoolSize, Conv2Filters);
		std::println("  Flatten:     {} neurons", 4 * 4 * Conv2Filters);
		std::println("  FC:			 {} neurons", FCLayerSize);
		std::println("  Dropout:     {:.1f}% rate", static_cast<double>(DropoutRate * 100));
		std::println("  Output:		 {} neurons (softmax)", OutputSize);
		std::printnl();

		network<sequential> neuralNetwork;
		
		// Layer 1: Convolution (28x28x1 -> 24x24x32)
		neuralNetwork << convolutional_layer<activation::relu>(ImageWidth, ImageHeight, Conv1KernelSize, 1, Conv1Filters);
		
		// Layer 2: Max Pooling (24x24x32 -> 12x12x32)
		neuralNetwork << max_pooling_layer<activation::identity>(24, 24, Conv1Filters, PoolSize);
		
		// Layer 3: Convolution (12x12x32 -> 8x8x64)
		neuralNetwork << convolutional_layer<activation::relu>(12, 12, Conv2KernelSize, Conv1Filters, Conv2Filters);
		
		// Layer 4: Max Pooling (8x8x64 -> 4x4x64)
		neuralNetwork << max_pooling_layer<activation::identity>(8, 8, Conv2Filters, PoolSize);
		
		// Layer 5: Fully Connected (1024 -> 128)
		neuralNetwork << fully_connected_layer<activation::relu>(4 * 4 * Conv2Filters, FCLayerSize);
		
		// Layer 6: Dropout for regularization
		neuralNetwork << dropout_layer(FCLayerSize, DropoutRate);
		
		// Layer 7: Output (128 -> 10)
		neuralNetwork << fully_connected_layer<activation::softmax>(FCLayerSize, OutputSize);

		// Use Adam optimizer with learning rate
		adam optimiser;
		optimiser.alpha = 0.001f;  // Learning rate

		// Track epochs
		size_t epochCount = 0;

		// Progress display for monitoring training
		progress_display progressDisplay(TrainingEpochs);

		auto onEpochComplete = [&]
		{
			++epochCount;
			++progressDisplay;

			const tiny_cnn::float_t loss = neuralNetwork.get_loss<mse>(trainingInputs, trainingOutputs);
			std::printnl();
			std::println("Epoch {}/{}: Loss = {:.6f}", epochCount, TrainingEpochs, loss);
			
			// Evaluate on a subset of test data every epoch for progress monitoring
			if (epochCount % 2 == 0)
			{
				const size_t testSubsetSize = std::min<size_t>(1000, testInputs.size());
				size_t correct = 0;
				
				for (size_t i = 0; i < testSubsetSize; ++i)
				{
					const vec_t prediction = neuralNetwork.predict(testInputs[i]);
					if (GetPredictedDigit(prediction) == GetExpectedDigit(testOutputs[i]))
					{
						++correct;
					}
				}
				
				const tiny_cnn::float_t accuracy = 100.0 * static_cast<tiny_cnn::float_t>(correct) / static_cast<tiny_cnn::float_t>(testSubsetSize);
				std::println("  Test accuracy (subset): {:.2f}% ({}/{})", static_cast<double>(accuracy), correct, testSubsetSize);
			}
		};

		auto onMinibatchComplete = [&] {};

		std::printnl();
		std::println("Starting training...");
		std::println("  Batch size: {}", BatchSize);
		std::println("  Epochs: {}", TrainingEpochs);
		std::println("  Learning rate: {:.4f}", optimiser.alpha);
		std::println("  Optimizer: Adam");
		std::printnl();

		// Train the network using MSE loss function
		// MSE is numerically stable and works well with softmax output
		neuralNetwork.fit<mse>(optimiser, trainingInputs, trainingOutputs, BatchSize, TrainingEpochs, onMinibatchComplete, onEpochComplete);

		// Evaluate final accuracy on the full test set
		std::printnl();
		std::println("Evaluating on full test set...");
		EvaluateAndPrintAccuracy(neuralNetwork, testInputs, testOutputs);

		// Let the user test the network
		InteractiveTest(neuralNetwork, testInputs, testOutputs);
	}
}
