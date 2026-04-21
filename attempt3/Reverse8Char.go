/*
 * This work is marked with CC0 1.0 Universal. 
 * To view a copy of this license, visit:
 * http://creativecommons.org
 *
 * No Copyright: The person who associated a work with this deed has dedicated 
 * the work to the public domain by waiving all of his or her rights to the work 
 * worldwide under copyright law, including all related and neighboring rights, 
 * to the extent allowed by law.
 */

// $ go mod init example.com/Reverse8Char/v2
// $ go mod tidy
// $ go build

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

type Neuron struct {
	Weights []float64
	Bias    float64
}

func sigmoid(x float64) float64          { return 1.0 / (1.0 + math.Exp(-x)) }
func sigmoidDerivative(s float64) float64 { return s * (1.0 - s) }

func (n *Neuron) Process(inputs []float64) float64 {
	var sum float64
	for i, input := range inputs {
		sum += input * n.Weights[i]
	}
	return sigmoid(sum + n.Bias)
}

func (n *Neuron) Train(inputs []float64, target float64, lr float64) float64 {
	prediction := n.Process(inputs)
	errorDelta := target - prediction
	gradient := errorDelta * sigmoidDerivative(prediction)

	for i := range n.Weights {
		n.Weights[i] += inputs[i] * gradient * lr
	}
	n.Bias += gradient * lr

    // Return absolute error for monitoring
	return math.Abs(errorDelta)
}

func stringToFloats(s string) []float64 {
	f := make([]float64, len(s))
	for i, r := range s { f[i] = float64(r) / 255.0 }
	return f
}

func printResults(network []*Neuron, inputs []float64) {
	result := ""
	for i := 0; i < 8; i++ {
		raw := network[i].Process(inputs)
		result += string(rune(int(raw*255.0 + 0.5)))
	}
	fmt.Printf("Input: abcdefgh -> Output: %s\n", result)
}

func main() {
	inputStr := "abcdefgh"
	targetStr := "hgfedcba"
	inputs := stringToFloats(inputStr)

	network := make([]*Neuron, 8)
	for i := range network {
		network[i] = &Neuron{Weights: make([]float64, 8), Bias: rand.Float64()}
	}

	// Channel to collect error rates (buffered to prevent blocking)
	errorChan := make(chan float64, 100)
	var wg sync.WaitGroup

	// 1. Start a "Monitor" goroutine to print errors from the channel
	go func() {
		count := 0
		var totalError float64
		for err := range errorChan {
			totalError += err
			count++
			if count%800 == 0 { // Print average error every 100 epochs (8 neurons * 100)
				fmt.Printf("Avg Error at step %d: %.5f\n", count/8, totalError/800)
				totalError = 0
			}
		}
	}()

	// 2. Launch Parallel Training
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(pos int) {
			defer wg.Done()
			target := float64(targetStr[pos]) / 255.0
			for e := 0; e < 5000; e++ {
				loss := network[pos].Train(inputs, target, 0.5)
				errorChan <- loss // Send loss to monitor
			}
		}(i)
	}

	wg.Wait()
	close(errorChan) // Closing channel stops the monitor goroutine

	fmt.Println("\nFinal Result:")
	printResults(network, inputs)
}
