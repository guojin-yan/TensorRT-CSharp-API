#ifndef EXCEPTION_STATUS_H
#define EXCEPTION_STATUS_H

// catch all exception
enum class ExceptionStatus : int {
	NotOccurred = 0,
	Occurred = 1,
	OccurredTRT = 2,
	OccurredCuda = 3
};

#endif // !EXCEPTION_STATUS_H



