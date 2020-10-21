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
package workerpool

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestWorkerpool(t *testing.T) {
	wp := New(map[string]int{"L": 10, "P": 0})
	var runningL int64
	var runningP int64
	for j := 0; j < 100; j++ {
		wp.Queue("P", func() error {
			atomic.AddInt64(&runningP, 1)
			for i := 0; i < 10; i++ {
				wp.Queue("L", func() error {
					atomic.AddInt64(&runningL, 1)
					time.Sleep(time.Millisecond * 1)
					atomic.AddInt64(&runningL, -1)
					return nil
				})
			}
			atomic.AddInt64(&runningP, -1)
			return nil
		})
	}
	var done int64
	pMoreThan10 := false
	lMoreThan10 := false
	doneLooking := &sync.WaitGroup{}
	doneLooking.Add(1)
	go func() {
		for atomic.LoadInt64(&done) == 0 {
			if atomic.LoadInt64(&runningP) > 10 {
				pMoreThan10 = true
			}
			if atomic.LoadInt64(&runningL) > 10 {
				lMoreThan10 = true
			}
		}
		doneLooking.Done()
	}()
	wp.Close("P")
	wp.Wait("P")
	wp.Close("L")
	wp.Wait("L")
	atomic.StoreInt64(&done, 1)
	doneLooking.Wait()
	if !pMoreThan10 {
		t.Errorf("Unlimited job queue didn't fill up!")
	}
	if lMoreThan10 {
		t.Errorf("Limited job queue didn't stick to its limits!")
	}
}
