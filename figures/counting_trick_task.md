# Task: Create a nested sampling counting trick visualization script

## Context
We have an existing nested sampling script `himmelblau_ns.py` that uses the BlackJAX nested sampling branch to create animated visualizations of the nested sampling algorithm on the Himmelblau function. We need to create a new script `himmelblau_ns_counting_trick.py` that demonstrates the "counting trick" aspect of how nested sampling estimates volumes.

## Requirements
1. **Base it directly on the existing `himmelblau_ns.py` script** - use the same BlackJAX nested sampling implementation, same parameters (200 live points), same setup
2. **Generate exactly 3 frames** to show the counting trick progression:
   - Frame 1: Initial setup with 200 live points, no likelihood contour drawn
   - Frame 2: After a few steps, showing the likelihood contour (dashed red line) and points classified as inside/outside/dead
   - Frame 3: After more steps, showing further contour shrinking with more dead points
3. **Use three distinct colors for point classification**:
   - **C0 (blue)** for live points inside the current likelihood contour
   - **C3 (red)** for live points outside the current likelihood contour  
   - **C5 (brown)** for dead points (use squares to distinguish from circles)
4. **Add titles to each frame** showing the counts: "Step X: Y inside, Z outside, W dead"
5. **Only draw the likelihood threshold contour** (dashed red line) after the first frame
6. **Keep all other aspects the same** as the original: same Himmelblau function, same plotting style, same contour levels, same figure size (3,3)

## Key insight
The counting trick shows how nested sampling estimates volume shrinkage by literally counting how many live points are inside vs outside each likelihood contour, leading to the volume estimate $V_{i+1} = V_i \times \frac{n-1}{n}$.

## Output
The script should be called `himmelblau_ns_counting_trick.py` and generate `himmelblau_ns_counting_trick.pdf`.