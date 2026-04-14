using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using JYPPX.TensorRtSharp.Internal.Fundamentals;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using JYPPX.TensorRtSharp.ExternalInterface;
namespace JYPPX.TensorRtSharp.Nvinfer
{
    /// <summary>
    /// ��־��Ϣ�¼�������
    /// Log message event arguments
    /// </summary>
    public class LogEventArgs : EventArgs
    {
        /// <summary>
        /// ��־�ȼ�
        /// Log severity level
        /// </summary>
        public LoggerSeverity Severity { get; set; }

        /// <summary>
        /// ��־��Ϣ
        /// Log message
        /// </summary>
        public string Message { get; set; }

        /// <summary>
        /// ��ʽ����־��Ϣ��������ʱ����ͼ���ǰ׺��
        /// Formatted log message (with timestamp and severity prefix)
        /// </summary>
        public string FormattedMessage { get; set; }

        /// <summary>
        /// �¼�������ʱ��
        /// Event timestamp
        /// </summary>
        public DateTime Timestamp { get; set; }

        public LogEventArgs(LoggerSeverity severity, string message, string formattedMessage)
        {
            Severity = severity;
            Message = message;
            FormattedMessage = formattedMessage;
            Timestamp = DateTime.Now;
        }
    }

    /// <summary>
    /// һ��������ģʽ����־��¼���������� TensorRT �ĵ������־ϵͳ���桢
    /// A singleton logger for interacting with the native TensorRT logging system.
    /// </summary>
    public class Logger
    {
        private IntPtr ptr;
        // ˽�о�̬ʵ����ȷ��Ψһ��
        // Private static instance (to ensure uniqueness).
        private static Logger _instance;
        // ˽�й��캯������ֹ�ⲿʵ����
        // Private constructor (to prevent external instantiation).
        private Logger()
        {
            InitHandleException.handler(
                NativeMethods.trtLogger_getTrtLogger(out ptr));
        }
        // ������̬������ȡ��һʵ��
        // Public static property to get the unique instance.
        /// <summary>
        /// ��ȡ Logger ��ȫ��Ψһʵ����
        /// Gets the global unique instance of the Logger class.
        /// </summary>
        /// <returns>Logger ������ʵ�� / The singleton instance of Logger.</returns>
        public static Logger Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new Logger();
                }
                return _instance;
            }
        }

        /// <summary>
        /// ��־��Ϣ�¼����������ڽ���ԭ����־��Ϣ
        /// Event triggered when a log message is received (raw message with severity)
        /// </summary>
        public event EventHandler<LogEventArgs> OnLog;

        /// <summary>
        /// �ڲ��ص��������� C++ ���ÿ��Ʊ��뱣��
        /// Internal callback function to prevent GC collection
        /// </summary>
        private LogCallbackFunctionV2 _internalCallbackV2;

        private bool _callbackRegistered = false;

        /// <summary>
        /// ��¼��־���������Լ��������ڴ�¼�����Ե���־���Ա��ԡ�
        /// Sets the minimum severity level for logging. Logs below this level will be ignored.
        /// </summary>
        /// <param name="threshold">��־�����͹������Լ� / The minimum severity level for logging.</param>
        public void SetThreshold(LoggerSeverity threshold)
        {
            NativeMethods.trtLogger_setThreshold(threshold);
        }

        /// <summary>
        /// ��ȡ��ǰ��¼��־���������Լ�
        /// Gets the current minimum severity level for logging.
        /// </summary>
        /// <returns>��ǰ��־�������Լ� / Current minimum severity level for logging.</returns>
        public LoggerSeverity GetThreshold()
        {
            NativeMethods.trtLogger_getThreshold(out LoggerSeverity threshold);
            return threshold;
        }

        /// <summary>
        /// ����һ���Զ���Ļص�����������־��Ϣ��Ĭ��ʹ��׼����������̨����
        /// Sets a custom callback function to handle log messages. Defaults to standard output if not set.
        /// </summary>
        /// <param name="callback">���ϸ�ʽ����־��Ϣ�Ļص����� / The callback function that receives formatted log messages.</param>
        public void SetCallback(LogCallbackFunction callback)
        {
            NativeMethods.trtLogger_setCallback(callback);
            _callbackRegistered = callback != null;
        }

        /// <summary>
        /// ����һ���Զ���Ļص����������� severity ����־��Ϣ
        /// Sets a custom callback function to handle log messages with severity information.
        /// </summary>
        /// <param name="callback">���� severity ����־��Ϣ�Ļص����� / The callback function that receives severity and message.</param>
        public void SetCallbackV2(LogCallbackFunctionV2 callback)
        {
            NativeMethods.trtLogger_setCallbackV2(callback);
            _callbackRegistered = callback != null;
        }

        /// <summary>
        /// �����¼������ƴ����ڽ���ԭ����־��Ϣ
        /// Enables event-based logging, triggering OnLog event for each log message.
        /// </summary>
        public void EnableEventLogging()
        {
            if (_callbackRegistered)
                return;

            _internalCallbackV2 = (severity, msgPtr) =>
            {
                string msg = Marshal.PtrToStringAnsi(msgPtr);
                var severityEnum = (LoggerSeverity)severity;
                string formattedMsg = FormatLogMessage(severityEnum, msg);
                
                OnLog?.Invoke(this, new LogEventArgs(severityEnum, msg, formattedMsg));
            };

            NativeMethods.trtLogger_setCallbackV2(_internalCallbackV2);
            _callbackRegistered = true;
        }

        /// <summary>
        /// ͣ���¼������ƣ��ָ�Ĭ����������
        /// Disables event-based logging and restores default output behavior.
        /// </summary>
        public void DisableEventLogging()
        {
            NativeMethods.trtLogger_setCallbackV2(null);
            _internalCallbackV2 = null;
            _callbackRegistered = false;
        }

        /// <summary>
        /// ��ʽ����־��Ϣ
        /// Formats a log message with timestamp and severity.
        /// </summary>
        private string FormatLogMessage(LoggerSeverity severity, string message)
        {
            string severityStr = severity switch
            {
                LoggerSeverity.kINTERNAL_ERROR => "FATAL",
                LoggerSeverity.kERROR => "ERROR",
                LoggerSeverity.kWARNING => "WARNING",
                LoggerSeverity.kINFO => "INFO",
                LoggerSeverity.kVERBOSE => "VERBOSE",
                _ => "UNKNOWN"
            };
            return $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] [{severityStr}] {message}";
        }

        /// <summary>
        /// ��¼һ��ָ������Լ���Ϣ��
        /// Logs a message with a specified severity level.
        /// </summary>
        /// <param name="level">��־������Լ� / The severity level of the log.</param>
        /// <param name="msg">Ҫ��¼���������� / The content of the message to log.</param>
        public void Log(LoggerSeverity level, string msg)
        {
            NativeMethods.trtLogger_log(level, msg);
        }

        /// <summary>
        /// ��¼һ�� VERBOSE�����ϸ��������Ϣ��
        /// Logs a VERBOSE level message.
        /// </summary>
        /// <param name="message">һ�����ϸ�ʽ�ַ���������¼������ / A composite format string that contains the text to log.</param>
        /// <param name="args">һ�����������������������ĸ�ʽ��Ķ��� / An object array that contains zero or more objects to format.</param>
        public void VERBOSE(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // ��ʽ�� / Format
                : message;  // �޲�����ʱֱ������ / Output directly when there are no arguments.
            Log(LoggerSeverity.kVERBOSE, formattedMsg);
        }

        /// <summary>
        /// ��¼һ�� INFO����Ϣ��������Ϣ��
        /// Logs an INFO level message.
        /// </summary>
        /// <param name="message">һ�����ϸ�ʽ�ַ���������¼������ / A composite format string that contains the text to log.</param>
        /// <param name="args">һ�����������������������ĸ�ʽ��Ķ��� / An object array that contains zero or more objects to format.</param>
        public void INFO(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // ��ʽ�� / Format
                : message;  // �޲�����ʱֱ������ / Output directly when there are no arguments.
            Log(LoggerSeverity.kINFO, formattedMsg);
        }

        /// <summary>
        /// ��¼һ�� WARNING�����棺����Ϣ��
        /// Logs a WARNING level message.
        /// </summary>
        /// <param name="message">һ�����ϸ�ʽ�ַ���������¼������ / A composite format string that contains the text to log.</param>
        /// <param name="args">һ�����������������������ĸ�ʽ��Ķ��� / An object array that contains zero or more objects to format.</param>
        public void WARNING(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // ��ʽ�� / Format
                : message;  // �޲�����ʱֱ������ / Output directly when there are no arguments.
            Log(LoggerSeverity.kWARNING, formattedMsg);
        }

        /// <summary>
        /// ��¼һ�� ERROR��������Ϣ��
        /// Logs an ERROR level message.
        /// </summary>
        /// <param name="message">һ�����ϸ�ʽ�ַ���������¼������ / A composite format string that contains the text to log.</param>
        /// <param name="args">һ�����������������������ĸ�ʽ��Ķ��� / An object array that contains zero or more objects to format.</param>
        public void ERROR(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // ��ʽ�� / Format
                : message;  // �޲�����ʱֱ������ / Output directly when there are no arguments.
            Log(LoggerSeverity.kERROR, formattedMsg);
        }

        /// <summary>
        /// ��¼һ�� INTERNAL_ERROR���ڲ������������Ϣ��
        /// Logs an INTERNAL_ERROR level message.
        /// </summary>
        /// <param name="message">һ�����ϸ�ʽ�ַ���������¼������ / A composite format string that contains the text to log.</param>
        /// <param name="args">һ�����������������������ĸ�ʽ��Ķ��� / An object array that contains zero or more objects to format.</param>
        public void INTERNAL_ERROR(string message, params object[] args)
        {
            string formattedMsg = args.Length > 0
                ? string.Format(message, args)  // ��ʽ�� / Format
                : message;  // �޲�����ʱֱ������ / Output directly when there are no arguments.
            Log(LoggerSeverity.kINTERNAL_ERROR, formattedMsg);
        }
    }
}
