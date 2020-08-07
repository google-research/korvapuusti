/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 */
package tfexamples

import (
	"testing"
)

func TestGetNumberFromStr(t *testing.T) {
	tests := []struct {
		desc  string
		input string
		want  float32
	}{
		{
			desc:  "Handle leading and trailing zeros",
			input: "0060.00",
			want:  60.0,
		},
		{
			desc:  "Handle leading zeros",
			input: "0050",
			want:  50.0,
		},
		{
			desc:  "Handle trailing zeros",
			input: "47.0",
			want:  47.0,
		},
		{
			desc:  "Handle decimal part",
			input: "47.998",
			want:  47.998,
		},
		{
			desc:  "Handle decimal part",
			input: "47.1",
			want:  47.1,
		},
	}

	for _, tc := range tests {
		if got, _ := getNumberFromStr(tc.input); got != tc.want {
			t.Errorf("%v: getNumberFromStr(%v) is unexpectedly %v", tc.desc, tc.input, got)
		}
	}
}

func TestExtractMetadataFile(t *testing.T) {
	tests := []struct {
		desc  string
		input string
		want1 []float32
		want2 []float32
	}{
		{
			desc:  "Handle leading and trailing zeros",
			input: "r000001_combined_00060.00Hz_-50.00dBFS+04800.00Hz_-60.00dBFS.wav",
			want1: []float32{60.0, 4800.0},
			want2: []float32{-50.0, -60.0},
		},
		{
			desc:  "Handle leading and trailing zeros and decimal points",
			input: "r000001_combined_00060.008Hz_-50.50dBFS+04800.99Hz_-60.00dBFS.wav",
			want1: []float32{60.008, 4800.99},
			want2: []float32{-50.5, -60.0},
		},
	}

	for _, tc := range tests {
		got1, got2, _ := extractMetadataFile(tc.input)
		for idx, got := range got1 {
			if got != tc.want1[idx] {
				t.Errorf("%v: extractMetadataFile(%v) is unexpectedly %v", tc.desc, tc.input, got)
			}
		}
		for idx, got := range got2 {
			if got != tc.want2[idx] {
				t.Errorf("%v: extractMetadataFile(%v) is unexpectedly %v", tc.desc, tc.input, got)
			}
		}
	}
}
