# Combined Video Update for Multi-Column Predictions

## Summary

Updated the multi-column prediction script to create a **single combined 3D scatter plot video** showing all target columns together, instead of separate videos for each column.

## Changes Made

### 1. Modified `generate_gino_predictions_multi_col.py`

#### Added Functions:
- **`create_combined_3d_scatter_plots()`**
  - Creates 3D scatter plots with all target columns in one figure
  - Layout: Rows = target columns, Columns = observations/predictions/error
  - Saves individual frames to `combined_3d_scatter_plots/` directory
  
- **`create_video_from_combined_scatter_plots()`**
  - Generates single video from combined plot frames
  - Output: `combined_3d_scatter_plots_video.mp4`
  - 10 fps, cycles through all samples

#### Updated Workflow:
- Removed per-column video generation
- Added combined video generation step
- Per-column analyses still include 2D plots but no longer include 3D scatter plots or videos

### 2. Updated `README_multi_col_predictions.md`

- Added description of combined video feature
- Updated directory structure to show new output files
- Added visual layout diagrams showing video structure
- Removed references to per-column 3D scatter plot videos

## New Output Structure

```
predictions_dir/
├── combined_3d_scatter_plots_video.mp4     # ⭐ Single video with all targets
├── combined_3d_scatter_plots/              # Individual frames
│   ├── combined_3d_scatter_sample_001.png
│   ├── combined_3d_scatter_sample_002.png
│   └── ...
├── first_timestep_all_columns.png
├── mass_concentration/                      # Per-column 2D analyses only
│   ├── predictions_vs_observations.png
│   ├── time_series_comparison.png
│   └── error_analysis.png
└── head/
    └── ... (same structure)
```

## Video Layout

### For 2 Target Columns (e.g., mass_concentration + head):
```
Row 1: [ mass_concentration Obs | mass_concentration Pred | Error ]
Row 2: [       head Obs          |       head Pred         | Error ]
```

### For 3 Target Columns (e.g., mass_concentration + head + pressure):
```
Row 1: [ mass_concentration Obs | mass_concentration Pred | Error ]
Row 2: [       head Obs          |       head Pred         | Error ]
Row 3: [     pressure Obs        |     pressure Pred       | Error ]
```

Each frame shows one sample with all target variables visualized together.

## Benefits

1. **Single unified view**: Compare all target columns simultaneously
2. **Easier comparison**: See how different variables behave spatially
3. **Reduced file clutter**: One video instead of multiple
4. **Better for presentations**: Comprehensive view in one file
5. **Smaller storage**: Fewer video files to manage

## Usage

No changes to the PBS script or command-line interface. Simply run:

```bash
qsub generate_gino_predictions_multi_col.pbs
```

The combined video will be generated automatically along with all other analyses.

## Backward Compatibility

- Per-column 2D analyses unchanged (scatter plots, time series, error analysis)
- Data files unchanged (predictions, targets, coordinates)
- All existing functionality preserved
- Only change: replaced per-column 3D videos with one combined video

## Technical Details

### Plot Dimensions
- Figure size: 18 inches wide × (6 × n_target_cols) inches tall
- Each subplot: 3D scatter plot with colorbar
- Font sizes optimized for readability in large multi-row layouts

### Performance
- Progress updates every 10 samples during plot generation
- Video encoding uses standard mp4v codec
- Frame rate: 10 fps (same as before)

### Color Scales
- Observations/Predictions: Same color scale per variable (vmin/vmax from observations)
- Error plots: Diverging colormap (RdBu_r) centered at 0
- Each variable uses its own appropriate range

## Files Modified

1. `generate_gino_predictions_multi_col.py` - Added combined video functions
2. `README_multi_col_predictions.md` - Updated documentation
3. `CHANGES_combined_video.md` - This file (change summary)

## Testing Recommendations

When you first run the updated script:
1. Check that `combined_3d_scatter_plots_video.mp4` is created
2. Verify video plays correctly and shows all target columns
3. Confirm individual frames are saved in `combined_3d_scatter_plots/`
4. Ensure per-column 2D analyses still work as expected
5. Check console output for progress messages

## Example Console Output

```
Creating combined 3D scatter plots for all target columns...
Creating combined 3D scatter plots for 150 samples with 2 target columns...
  Created 10/150 combined plots
  Created 20/150 combined plots
  ...
Combined 3D scatter plots saved to: .../combined_3d_scatter_plots/

Creating video from combined 3D scatter plots...
Found 150 combined scatter plot images
Creating combined video frames: 100%|████████| 150/150 [00:05<00:00, 28.32it/s]
Combined video saved to: .../combined_3d_scatter_plots_video.mp4
Video details: 150 frames at 10 fps
```

## Support

If you encounter any issues:
1. Check that all target columns are specified correctly
2. Verify sufficient disk space for video generation
3. Ensure matplotlib and cv2 are properly installed
4. Check that predictions were generated successfully

---

**Date**: 2025-10-01  
**Version**: 1.0  
**Status**: Ready for use

