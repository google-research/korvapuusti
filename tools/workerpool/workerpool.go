/* workerpool contains code to run a limited number of error handling goroutines concurrently.
 *
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
	"fmt"
	"sync"
)

// MultiErr contains multiple errors.
type MultiErr []error

// Error returns a string representation of the multi error.
func (m MultiErr) Error() string {
	return fmt.Sprint([]error(m))
}

// WorkerPool runs a limited number of error handling goroutines concurrently.
type WorkerPool struct {
	queue  chan func() error
	errors chan error
}

// Go will run the function.
func (w *WorkerPool) Go(f func() error) {
	w.queue <- f
}

// Wait stops accepting jobs, and waits for queue to be closed and all submitted jobs to finish and returns the errors.
func (w *WorkerPool) Wait() error {
	close(w.queue)
	me := MultiErr{}
	for err := range w.errors {
		if err != nil {
			me = append(me, err)
		}
	}
	if len(me) == 0 {
		return nil
	}
	return me
}

// New returns a new worker pool.
func New(concurrency int) *WorkerPool {
	w := &WorkerPool{
		queue:  make(chan func() error),
		errors: make(chan error),
	}

	go func() {
		wg := &sync.WaitGroup{}
		tickets := make(chan struct{}, concurrency)
		for jobVar := range w.queue {
			job := jobVar
			if concurrency > 0 {
				tickets <- struct{}{}
			}
			wg.Add(1)
			go func() {
				err := job()
				if concurrency > 0 {
					<-tickets
				}
				w.errors <- err
				wg.Done()
			}()
		}
		wg.Wait()
		close(w.errors)
	}()
	return w
}
