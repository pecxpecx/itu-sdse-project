// main.go
package main

import (
	"context"
	"fmt"
	"dagger.io/dagger" 
)

type Dagger struct{}

func (m *Dagger) MLOpsPipeline(ctx context.Context, source *Directory) *Directory {
	
	// Define Base Environment and Install Dependencies
	pythonBase := dag.Container().From("python:3.11-slim")
	app := pythonBase.WithMountedDirectory("/app", source).WithWorkdir("/app")

	// Install Python dependencies
	app = app.WithExec([]string{"pip", "install", "-r", "requirements.txt"})

	// --- 1. Data Preparation Stage ---
	fmt.Println("--- Stage 1: Running Data Preparation (src/data_prep.py) ---")
	
	dataPrepContainer := app.WithExec([]string{"python", "src/data_prep.py"})
	pipelineState := dataPrepContainer
	
	// --- 2. Model Training Stage ---
	fmt.Println("--- Stage 2: Running Model Training (src/train.py) ---")

	trainContainer := pipelineState.WithExec([]string{"python", "src/train.py"})
	pipelineState = trainContainer

	// --- 3. Model Deployment Stage ---
	fmt.Println("--- Stage 3: Running Model Deployment (src/deploy.py) ---")

	deployContainer := pipelineState.WithExec([]string{"python", "src/deploy.py"})
	pipelineState = deployContainer
	
	// --- 4. Export Final Artifacts ---

	// The professor requires all history (mlruns/) and the model artifact (artifacts/)
	artifactsDir := pipelineState.Directory("artifacts")
	mlrunsDir := pipelineState.Directory("mlruns")

	outputDir := dag.Directory()
	outputDir = outputDir.WithDirectory("artifacts", artifactsDir)
	outputDir = outputDir.WithDirectory("mlruns", mlrunsDir)
	
	fmt.Println("--- Pipeline Complete. Final artifacts are ready for export. ---")
	
	return outputDir
}