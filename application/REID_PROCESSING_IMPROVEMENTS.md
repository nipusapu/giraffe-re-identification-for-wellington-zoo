# ReID Processing Improvements

## Overview

This document outlines the improvements made to the ReID (Re-identification) processing system to implement a "download-process-delete" workflow and improve local development experience.

## Key Changes

### 1. S3 Configuration

- **Default S3 disabled**: Changed `AWS_USE_S3` default from `True` to `False` for local development
- **Fallback storage**: Added fallback to local file storage when S3 is disabled
- **Environment configuration**: Created `env.example` file for easy S3 setup

### 2. Enhanced File Handling

- **Smart storage detection**: `_ensure_local()` function now detects whether S3 or local storage is being used
- **Local file support**: When S3 is disabled, the system looks for files in the local `MEDIA_ROOT` directory
- **Graceful fallbacks**: Better error handling for both S3 and local storage scenarios

### 3. Temporary File Management

- **Dedicated temp directory**: Uses `reid_processing` subdirectory for better organization
- **Configurable location**: `REID_TEMP_DIR` setting allows custom temp directory paths
- **Automatic cleanup**: `cleanup_temp_files()` task removes orphaned temporary files
- **Safe cleanup**: `_cleanup_temp_files()` helper function handles multiple file cleanup

## Workflow

### When S3 is Enabled:

1. **Download**: Image is downloaded from S3 to local server
2. **Process**: Detection and ReID processing runs on local copy
3. **Upload**: Results are uploaded back to S3
4. **Cleanup**: All temporary local files are deleted

### When S3 is Disabled (Local Development):

1. **Local Access**: Image is accessed directly from local `MEDIA_ROOT` directory
2. **Process**: Detection and ReID processing runs on local file
3. **Save**: Results are saved to local storage
4. **Cleanup**: Temporary processing files are cleaned up

## Configuration

### Environment Variables

```bash
# S3 Storage (set to False for local development)
AWS_USE_S3=False

# When AWS_USE_S3=True, also set:
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_STORAGE_BUCKET_NAME=your-bucket-name
AWS_S3_REGION_NAME=ap-southeast-2

# File Processing
REID_TEMP_DIR=  # Custom temp dir, or leave empty for system default
REID_CLEANUP_INTERVAL=3600  # Cleanup every hour (seconds)
```

### Quick Setup for Local Development

1. Copy `env.example` to `.env` (if you want to customize settings)
2. Ensure `AWS_USE_S3=False` (default)
3. Place test images in the `media/` directory
4. Start the application

## File Structure

```
application/
├── media/                    # Local file storage (when S3 disabled)
│   └── images.jpg
├── reid_processing/         # Temporary processing directory
│   ├── temp_*.jpg          # Downloaded/cropped images
│   └── ...
├── config/
│   └── settings.py         # Updated with S3 toggle
├── reid/
│   └── tasks.py            # Enhanced file handling
└── env.example              # Configuration template
```

## Benefits

### For Development:

- **No S3 credentials required** for local testing
- **Faster processing** with local files
- **Easier debugging** with local file access
- **Configurable storage** via environment variables

### For Production:

- **Flexible deployment** - can use S3 or local storage
- **Better error handling** for storage failures
- **Automatic cleanup** prevents disk space issues
- **Maintains S3 workflow** when needed

## Monitoring

### Logs to Watch:

- `"Downloaded image from S3 to local"` - S3 downloads
- `"Found image in local media directory"` - Local file access
- `"Cleaned up temporary file"` - Cleanup operations
- `"Failed to download image from S3"` - S3 errors

### Health Checks:

- Monitor `reid_processing/` directory size
- Check Celery task success rates
- Verify storage backend configuration

## Troubleshooting

### Common Issues:

1. **403 Forbidden S3 Error**:

   - Check `AWS_USE_S3` setting
   - Verify S3 credentials and permissions
   - Set `AWS_USE_S3=False` for local development

2. **File Not Found**:

   - Ensure images exist in `media/` directory (local mode)
   - Check S3 bucket and key names (S3 mode)
   - Verify file paths in database

3. **Temp Directory Issues**:
   - Check `REID_TEMP_DIR` setting
   - Ensure write permissions
   - Monitor disk space

### Debug Commands:

```bash
# Check storage backend
python manage.py shell -c "from django.core.files.storage import default_storage; print(type(default_storage))"

# List temp directory contents
ls -la reid_processing/

# Check environment variables
python manage.py shell -c "from django.conf import settings; print(f'AWS_USE_S3: {getattr(settings, \"AWS_USE_S3\", None)}')"
```

## Migration Notes

### From Previous Version:

- **No breaking changes** to existing API
- **Automatic fallback** to local storage when S3 is disabled
- **Enhanced error handling** provides better debugging information
- **Backward compatible** with existing S3 workflows

### Testing:

1. **Test local mode**: Set `AWS_USE_S3=False` and place images in `media/`
2. **Test S3 mode**: Set `AWS_USE_S3=True` with valid credentials
3. **Verify cleanup**: Check that temporary files are removed after processing

## Future Enhancements

### Potential Improvements:

- **Storage backend detection** at runtime
- **Configurable cleanup policies** (age-based, size-based)
- **Storage metrics** and monitoring
- **Multi-region S3 support**
- **Local storage compression** options
