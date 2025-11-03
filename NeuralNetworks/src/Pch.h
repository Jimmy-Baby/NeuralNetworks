#pragma once

#include <cmath>
#include <bitset>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <format>
#include <vector>
#include <functional>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <ostream>
#include <string>
#include <optional>
#include <string_view>
#include <print>

namespace std
{
	// Custom 0 argument print with newline while the 0 argument version of std::println is spotty across compilers
	inline void printnl()
	{
		print(cout, "\n");
	}
}

// TinyDNN
#include "tiny_cnn/tiny_cnn.h"