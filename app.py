"""
Flask web application for ITHILDIN wing analysis.

This module provides a web interface for:
- Uploading wing images for analysis
- Processing images through the ITHILDIN pipeline
- Visualizing predictions and downloading results
- Accessing example data and documentation
- Selecting insect family for analysis (mosquito, drosophila, tsetse)
"""

import json
import os
import tempfile
import shutil
import threading
import time
import uuid
from zipfile import ZipFile

import pandas as pd
from flask import Flask, render_template, request, send_file, session, make_response
from rembg import new_session

import main
import utils
from config_loader import AVAILABLE_FAMILIES

# Initialize background removal session
bgremove_session = new_session()

# Flask app configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["STATIC_FOLDER"] = "static"
app.config["REQUESTS"] = "static/requests"
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "Apfelkuchen")

# How long (seconds) to keep temporary (non-consented) sessions before auto-deletion
TEMP_SESSION_TTL = 600  # 10 minutes


def _cleanup_expired_sessions():
    """Background thread: delete temporary request directories older than TEMP_SESSION_TTL."""
    while True:
        time.sleep(60)  # check every minute
        try:
            base_request_path = os.path.join(BASE_DIR, "static", "requests")
            if not os.path.isdir(base_request_path):
                continue
            now = time.time()
            for entry in os.scandir(base_request_path):
                if not entry.is_dir() or not entry.name.startswith("request_"):
                    continue
                meta_path = os.path.join(entry.path, "meta.json")
                if not os.path.exists(meta_path):
                    continue
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    if not meta.get("allow_storage", True):
                        age = now - meta.get("created_at", now)
                        if age >= TEMP_SESSION_TTL:
                            shutil.rmtree(entry.path, ignore_errors=True)
                            app.logger.info(f"Auto-deleted expired temp session: {entry.name}")
                except Exception as e:
                    app.logger.warning(f"Cleanup error for {entry.name}: {e}")
        except Exception as e:
            app.logger.warning(f"Background cleanup error: {e}")


_cleanup_thread = threading.Thread(target=_cleanup_expired_sessions, daemon=True)
_cleanup_thread.start()


def get_identifier():
    """
    Generate a unique identifier for each user request.

    Returns:
        str: Random 10-character identifier (UUID prefix)
    """
    return str(uuid.uuid4())[:11].replace("-", "")


@app.route("/search_session", methods=["POST"])
def search_session():
    """
    Search for a previous session by ID and display results.
    
    Returns:
        str: Rendered predictions page HTML if found, or redirect to home with error
    """
    session_id = request.form.get("session_id", "").strip()
    
    if not session_id:
        return render_template("upload.html", 
                             families=list(AVAILABLE_FAMILIES.keys()),
                             error="Please enter a session ID"), 400

    # Validate session_id to only allow safe alphanumeric characters (prevents path traversal)
    if not session_id.isalnum() or len(session_id) > 64:
        return render_template("upload.html",
                             families=list(AVAILABLE_FAMILIES.keys()),
                             error="Invalid session ID format. Please check the ID and try again."), 400
    
    # Build path to potential session
    base_request_path = os.path.join(app.root_path, "static", "requests")
    request_path = os.path.join(base_request_path, f"request_{session_id}")
    csv_file_path = os.path.join(request_path, f"coordinates_{session_id}.csv")
    
    # Check if the session exists
    if not os.path.exists(csv_file_path):
        return render_template("upload.html", 
                             families=list(AVAILABLE_FAMILIES.keys()),
                             error=f"Session ID '{session_id}' not found. Please check the ID and try again."), 404
    
    # Load the CSV file
    try:
        prediction_df = pd.read_csv(csv_file_path, sep=";")
        prediction_dict = prediction_df.to_dict(orient="records")
        
        # Determine family and has_classifier from the data
        # Try to detect from path or use default
        family = "mosquito"  # Default
        has_classifier = True  # Default for mosquito
        
        # Try to detect family from available columns
        if "CNN_Predicted_Taxa" not in prediction_df.columns:
            has_classifier = False
            
        # Update session
        session["identifier"] = session_id
        session["family"] = family
        session["request_path"] = os.path.join(app.config["REQUESTS"], f"request_{session_id}")
        session["request_path_processed"] = os.path.join(session["request_path"], "processed")

        # Read consent from stored metadata if available
        meta_path = os.path.join(request_path, "meta.json")
        allow_storage = True  # default: treat restored sessions as consented
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as mf:
                    allow_storage = json.load(mf).get("allow_storage", True)
            except Exception:
                pass
        session["allow_storage"] = allow_storage

        return render_template(
            "predictions.html", 
            predictions=prediction_dict, 
            request=session_id,
            family=family,
            has_classifier=has_classifier,
            allow_storage=allow_storage,
            upload_time="N/A",
            processing_time="N/A",
            total_time="N/A",
            num_images=len(prediction_dict)
        )
    except Exception as e:
        app.logger.error(f"Error loading session {session_id}: {e}")
        return render_template("upload.html", 
                             families=list(AVAILABLE_FAMILIES.keys()),
                             error=f"Error loading session: {str(e)}"), 500


@app.route("/")
def start():
    """
    Render the main upload page.

    Returns:
        str: Rendered HTML template for file upload interface
    """
    return render_template("upload.html", families=list(AVAILABLE_FAMILIES.keys()), error=None)


@app.route("/upload_folder", methods=["POST"])
def upload_folder():
    """
    Handle batch upload of wing images and run complete analysis pipeline.

    This endpoint:
    1. Creates a unique request directory for storing results
    2. Processes each uploaded image through ITHILDIN pipeline
    3. Generates landmark predictions using LDA
    4. Saves results to CSV and renders prediction page
    5. Tracks upload and processing time for user feedback

    Returns:
        str: Rendered predictions page HTML, or error message with status code

    Expected form data:
        file: Multiple image files (.jpg, .jpeg, .png, .tif, .tiff)
        family: Selected insect family (mosquito, drosophila, tsetse)
    """
    # Track start time for progress
    upload_start_time = time.time()
    
    # Read long-term storage consent (checkbox sends "on" when checked)
    allow_storage = request.form.get("allow_long_term_storage", "off") == "on"
    session["allow_storage"] = allow_storage

    # Get selected family (default to mosquito for backward compatibility)
    selected_family = request.form.get("family", "mosquito")
    if selected_family not in AVAILABLE_FAMILIES:
        return "Invalid insect family selection.", 400
    
    # Store family selection in session
    session["family"] = selected_family
    
    # Generate unique identifier for this request
    session["identifier"] = get_identifier()

    # Set up directory structure
    base_request_path = os.path.join(app.root_path, "static", "requests")
    request_id = f"request_{session['identifier']}"
    session["request_path"] = os.path.join(base_request_path, request_id)
    session["request_path_processed"] = os.path.join(
        session["request_path"], "processed"
    )

    # Create directories with error handling
    try:
        os.makedirs(session["request_path"], exist_ok=True)
        os.makedirs(session["request_path_processed"], exist_ok=True)
    except PermissionError as e:
        app.logger.error(f"Permission error: {e}")
        return "Server configuration issue: Unable to create necessary directories.", 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return "An unexpected error occurred.", 500

    # Write metadata for background cleanup
    meta_path = os.path.join(session["request_path"], "meta.json")
    with open(meta_path, "w") as f:
        json.dump({"allow_storage": allow_storage, "created_at": time.time()}, f)

    # Update session paths to use relative URLs
    session["request_path"] = os.path.join(
        app.config["REQUESTS"], f"request_{session['identifier']}"
    )
    session["request_path_processed"] = os.path.join(
        session["request_path"], "processed"
    )

    # Process uploaded files
    files = request.files.getlist("file")
    # Filter for valid image extensions
    allowed_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    files = [file for file in files if file.filename.endswith(allowed_extensions)]

    # Track upload completion time
    upload_time = time.time() - upload_start_time
    
    # Track processing start time
    processing_start_time = time.time()

    # Extract filenames and create output paths
    file_name_list = [file.filename.split(".")[0] for file in files]
    processed_file_name_list = [
        os.path.join(session["request_path_processed"], filename)
        for filename in file_name_list
    ]

    # Run ITHILDIN prediction pipeline on each image with family parameter
    # Collect timing information
    all_timings = []
    for i, file in enumerate(files):
        timing_info = {}
        main.run_prediction(file, save_path=processed_file_name_list[i], family=selected_family, timing_info=timing_info)
        all_timings.append(timing_info)
    
    # Track total processing time
    processing_time = time.time() - processing_start_time
    total_time = time.time() - upload_start_time
    
    # Calculate average timing for each step
    avg_timings = {}
    if all_timings:
        for key in all_timings[0].keys():
            avg_timings[key] = sum(t.get(key, 0) for t in all_timings) / len(all_timings)
    
    # Log timing information for performance analysis
    app.logger.info(f"Processing complete for session {session['identifier']}")
    app.logger.info(f"Number of images: {len(files)}")
    app.logger.info(f"Average timings per image: {avg_timings}")
    app.logger.info(f"Total processing time: {processing_time:.2f}s")

    # Generate dataframe from prediction results
    prediction_df = utils.json_to_dataframe(session["request_path_processed"], semilandmark=True, family=session["family"], with_lm_predictions=False, coordinate_type="unscaled")
    
    # Select relevant columns for output (conditionally include CNN columns)
    has_classifier = AVAILABLE_FAMILIES[selected_family]["has_classification"]

    # Add landmark-based species predictions
    prediction_df, prediction_lm_df, prediction_slm_df = main.get_landmark_predictions(prediction_df, session, has_classifier=has_classifier)

    # Update the main dataframe with the landmark predictions
    prediction_df = utils.json_to_dataframe(session["request_path_processed"], semilandmark=True, family=session["family"], with_lm_predictions=has_classifier, coordinate_type="scaled")

    # Prepare data for download and rendering
    main.prepare_download(prediction_df, prediction_lm_df, prediction_slm_df, has_classifier, session)

    # Prepare data for template
    prediction_dict = prediction_df.to_dict(orient="records")
    title = str(session["identifier"])

    return render_template(
        "predictions.html", 
        predictions=prediction_dict, 
        request=title,
        family=selected_family,
        has_classifier=has_classifier,
        allow_storage=allow_storage,
        upload_time=f"{upload_time:.2f}",
        processing_time=f"{processing_time:.2f}",
        total_time=f"{total_time:.2f}",
        num_images=len(files)
    )


@app.route("/cleanup_session", methods=["POST"])
def cleanup_session():
    """
    Delete uploaded images for the current session when the user has not
    consented to long-term storage.

    Called automatically via navigator.sendBeacon when the user leaves the
    predictions page.  Only image files inside the ``processed/`` sub-directory
    are removed; CSV / TPS result files are kept so the user can still
    retrieve them via the session-search feature.

    Returns:
        Response: 204 No Content on success, 400 if session data is missing.
    """
    identifier = session.get("identifier")
    allow_storage = session.get("allow_storage", True)

    if not identifier or identifier == "example":
        return make_response("", 204)

    if not allow_storage:
        request_dir = os.path.join(app.root_path, "static", "requests", f"request_{identifier}")
        processed_dir = os.path.join(request_dir, "processed")
        if os.path.isdir(processed_dir):
            shutil.rmtree(processed_dir, ignore_errors=True)
            app.logger.info(f"Deleted processed images for temp session: {identifier}")

    return make_response("", 204)


@app.route("/get_example")
def get_example():
    """
    Display example predictions without processing.

    Loads pre-computed example results and displays them in the
    predictions view for demonstration purposes.

    Returns:
        str: Rendered predictions page with example data
    """
    session["identifier"] = "example"
    session["family"] = "mosquito"  

    example_path = os.path.join(
        app.config["STATIC_FOLDER"], "example", "coordinates_example.csv"
    )

    prediction_dict = pd.read_csv(example_path, sep=";").to_dict(orient="records")

    return render_template(
        "predictions.html", 
        predictions=prediction_dict, 
        request="Example",
        family="mosquito",
        has_classifier=True,
        allow_storage=True,
        upload_time="0.00",
        processing_time="0.00",
        total_time="0.00",
        num_images=len(prediction_dict)
    )


@app.route("/display_pdf")
def display_pdf():
    """
    Serve the mosquito wing removal guide PDF.

    Returns:
        file: PDF file for in-browser display
    """
    pdf_path = os.path.join(
        app.config["STATIC_FOLDER"], "guide", "ConVector_MosquitoWingRemovalGuide.pdf"
    )
    return send_file(pdf_path, as_attachment=False)


@app.route("/display_other_pdf")
def display_other_pdf():
    """
    Serve the species labels guide PDF.

    Returns:
        file: PDF file for in-browser display
    """
    pdf_path = os.path.join(app.config["STATIC_FOLDER"], "guide", "Species.pdf")
    return send_file(pdf_path, as_attachment=False)


@app.route("/download_csv", methods=["GET"])
def download_csv():
    """
    Download predictions as CSV file.

    Returns the predictions CSV for either the current session or
    the example dataset, depending on the session identifier.

    Returns:
        file: CSV file as download attachment
    """
    if session["identifier"] == "example":
        csv_file_path = os.path.join(
            app.config["STATIC_FOLDER"], "example", "coordinates_example.csv"
        )
    else:
        csv_file_path = os.path.join(
            session["request_path"], f"coordinates_{session['identifier']}.csv"
        )
    return send_file(csv_file_path, as_attachment=True)


@app.route("/download_tps", methods=["GET"])
def download_tps():
    """
    Download raw coordinates as TPS file.

    Returns the TPS file for either the current session or
    the example dataset, depending on the session identifier.

    Returns:
        file: TPS file as download attachment
    """
    # Copy tps from analysis folder to request folder for download
    analysis_tps = os.path.join(app.root_path, "analysis", "temp", "input.tps")

    if session["identifier"] == "example":
        tps_file_path = os.path.join(
            app.config["STATIC_FOLDER"], "example", "coordinates_example.tps"
        )
    else:
        tps_file_path = os.path.join(
            session["request_path"], f"coordinates_{session['identifier']}_semi.tps"
        )

    shutil.copy(analysis_tps, tps_file_path)

    return send_file(tps_file_path, as_attachment=True)


@app.route("/download_folder", methods=["GET"])
def download_folder():
    """
    Download complete analysis results as ZIP archive.

    Packages all prediction files (JSON, images, CSV) from the session
    into a single ZIP file for download.

    Returns:
        file: ZIP archive containing all analysis results

    Note:
        Temporary ZIP file is automatically cleaned up after sending
    """
    if session["identifier"] == "example":
        folder_path = os.path.join(app.config["STATIC_FOLDER"], "example")
    else:
        folder_path = session["request_path"]

    zip_file_name = f"request_{session['identifier']}.zip"
    zip_path = os.path.join(tempfile.gettempdir(), zip_file_name)

    try:
        with ZipFile(zip_path, "w") as zipObj:
            # Get the root directory name to exclude from the zip archive
            root_dir = os.path.basename(os.path.normpath(folder_path))

            # Iterate over all files in directory tree
            for folderName, subfolders, filenames in os.walk(folder_path):
                for filename in filenames:
                    # Create complete filepath of file
                    filePath = os.path.join(folderName, filename)

                    # Get relative path excluding root directory
                    arcname = os.path.relpath(filePath, folder_path).replace(
                        root_dir + os.sep, "", 1
                    )

                    # Add file to zip with relative path
                    zipObj.write(filePath, arcname=arcname)

        # Send the zipped file
        return send_file(zip_path, as_attachment=True)
    finally:
        # Clean up temporary zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)


if __name__ == "__main__":
    app.run(debug=False)