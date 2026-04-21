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
	"time"
)

type Neuron struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

func sigmoid(x float64) float64           { return 1.0 / (1.0 + math.Exp(-x)) }
func sigmoidDerivative(s float64) float64  { return s * (1.0 - s) }

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
	return math.Abs(errorDelta)
}

// Data helpers
func stringToFloats(s string) []float64 {
	f := make([]float64, len(s))
	for i, r := range s { f[i] = float64(r) / 255.0 }
	return f
}

func generateRandomString(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyz"
	b := make([]byte, n)
	for i := range b { b[i] = letters[rand.Intn(len(letters))] }
	return string(b)
}

func reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	const filename = "weights.json"
	var network []*Neuron

	if _, err := os.Stat(filename); err == nil {
		fmt.Println("Loading trained weights...")
		data, _ := os.ReadFile(filename)
		json.Unmarshal(data, &network)
	} else {
		fmt.Println("Training for Generalization...")
		network = make([]*Neuron, 8)
		for i := range network {
			network[i] = &Neuron{Weights: make([]float64, 8), Bias: rand.Float64()}
			for j := range network[i].Weights { network[i].Weights[j] = rand.Float64() }
		}

		var wg sync.WaitGroup
		for i := 0; i < 8; i++ {
			wg.Add(1)
			go func(pos int) {
				defer wg.Done()
				// Train on 50,000 random samples to learn the pattern, not just one string
				for e := 0; e < 50000; e++ {
					s := generateRandomString(8)
					target := float64(reverse(s)[pos]) / 255.0
					network[pos].Train(stringToFloats(s), target, 0.3)
				}
			}(i)
		}
		wg.Wait()
		
		d, _ := json.Marshal(network)
		os.WriteFile(filename, d, 0644)
		fmt.Println("Training complete.")
	}

	// VALIDATION: Test on a completely new string
	testStr := "onionsys" // Not "abcdefgh"
	fmt.Printf("\nValidation Input:  %s\n", testStr)
	
	result := ""
	for i := 0; i < 8; i++ {
		val := network[i].Process(stringToFloats(testStr))
		result += string(rune(int(val*255.0 + 0.5)))
	}
	fmt.Printf("Neural Network Prediction: %s\n", result)
}
