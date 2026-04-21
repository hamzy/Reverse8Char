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
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
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

// SaveWeights writes the network configuration to a JSON file
func SaveWeights(filename string, network []*Neuron) error {
	data, err := json.MarshalIndent(network, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

// LoadWeights reads the network configuration from a JSON file
func LoadWeights(filename string) ([]*Neuron, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	var network []*Neuron
	err = json.Unmarshal(data, &network)
	return network, err
}

func main() {
	const filename = "weights.json"
	inputStr := "abcdefgh"
	targetStr := "hgfedcba"
	inputs := stringToFloats(inputStr)

	var network []*Neuron
	var err error

	// 1. Try to load existing weights
	if _, err = os.Stat(filename); err == nil {
		fmt.Println("Found existing weights. Loading...")
		network, err = LoadWeights(filename)
		if err != nil {
			fmt.Printf("Error loading weights: %v\n", err)
			return
		}
	} else {
		// 2. If no weights exist, initialize and train
		fmt.Println("No weights found. Starting parallel training...")
		network = make([]*Neuron, 8)
		for i := range network {
			network[i] = &Neuron{Weights: make([]float64, 8), Bias: rand.Float64()}
			for j := range network[i].Weights {
				network[i].Weights[j] = rand.Float64()
			}
		}

		errorChan := make(chan float64, 100)
		doneMonitor := make(chan bool)
		var wg sync.WaitGroup

		// Monitor goroutine
		go func() {
			count := 0
			var totalError float64
			for errVal := range errorChan {
				totalError += errVal
				count++
				if count%800 == 0 {
					fmt.Printf("Avg Error at epoch %d: %.5f\n", count/8, totalError/800)
					totalError = 0
				}
			}
			doneMonitor <- true
		}()

		// Parallel Training
		for i := 0; i < 8; i++ {
			wg.Add(1)
			go func(pos int) {
				defer wg.Done()
				target := float64(targetStr[pos]) / 255.0
				for e := 0; e < 10000; e++ {
					loss := network[pos].Train(inputs, target, 0.5)
					errorChan <- loss
				}
			}(i)
		}

		wg.Wait()
		close(errorChan)
		<-doneMonitor

		// 3. Save the newly trained weights
		err = SaveWeights(filename, network)
		if err != nil {
			fmt.Printf("Error saving weights: %v\n", err)
		} else {
			fmt.Println("Training complete and weights saved.")
		}
	}

	// 4. Final Output
	fmt.Println("\n--- Result ---")
	printResults(network, inputs)
}
