rem Bypass "Terminate Batch Job" prompt.
if "%~1"=="-FIXED_CTRL_C" (
   REM Remove the -FIXED_CTRL_C parameter
   SHIFT
) ELSE (
   REM Run the batch with <NUL and -FIXED_CTRL_C
   CALL <NUL %0 -FIXED_CTRL_C %*
   GOTO :EOF
)


@REM SQLALCHEMY_TRACK_MODIFICATIONS =false
@REM set FLASK_DEBUG=true
@REM set FLASK_APP=application && flask run



@REM cmd /K



@REM For MACOSX
export FLASK_APP=application && flask run --debug

flask run --debug

