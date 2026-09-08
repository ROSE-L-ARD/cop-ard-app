# Copyright 2024 European Union
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sarpy Readers Module Initialization

This module configures the Sarpy library to recognize and use custom Sentinel-1
readers (EOPF and SAFE formats) developed within the `cop_ard_app` package.

It imports the necessary Sarpy converter module and the custom reader modules,
then registers the custom "is_a" functions with Sarpy's opener mechanism.
This allows Sarpy's generic `open` function to automatically detect and use
these custom readers when given a path to a corresponding Sentinel-1 product.
"""

# Import the Sarpy complex data converter utility
# Import the custom Sarpy-compatible reader for Sentinel-1 EOPF files
import sarpy.io.complex.converter

import cop_ard_app.readers.sarpy_rosel as sarpy_rosel
import cop_ard_app.readers.sarpy_sentinel_safe as sarpy_safe

# This allows Sarpy to recognize individual SAFE measurement files.
sarpy.io.complex.converter.register_opener(sarpy_safe.is_a)
# This allows Sarpy to recognize individual ROSEL measurement files.
sarpy.io.complex.converter.register_opener(sarpy_rosel.is_a)
# This allows Sarpy to recognize individual SAOCOM products.
