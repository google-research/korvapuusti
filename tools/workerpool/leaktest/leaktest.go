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
package main

import (
	"runtime"

	"github.com/cheggaaa/pb"
	"github.com/google-research/korvapuusti/tools/workerpool"
)

func main() {
	wp := workerpool.New(map[string]int{"P": 0, "L": runtime.NumCPU()})
	jobs := 10000
	bar := pb.StartNew(jobs).Prefix("Running")
	wp.Queue("P", func() error {
		for i := 0; i < jobs; i++ {
			wp.Queue("L", func() error {
				a := make([]byte, 2<<24)
				for j := 0; j < 100; j++ {
					for idx := range a {
						a[idx] = a[idx] + byte(idx)
					}
				}
				bar.Increment()
				return nil
			})
		}
		wp.Close("L")
		return nil
	})
	wp.Close("P")
	if err := wp.WaitAll(); err != nil {
		panic(err)
	}
	bar.Finish()
}
