using JYPPX.TensorRtSharp.Exceptions;
using JYPPX.TensorRtSharp.ExternalInterface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JYPPX.TensorRtSharp.Internal.PInvoke
{

    public static class PInvokeHelper
    {
        /// <summary>
        /// Checks whether PInvoke functions can be called
        /// </summary>
        public static void TryPInvoke()
        {
            try
            {
                var size = NativeMethods.InitNvinfer();
            }
            catch (DllNotFoundException e)
            {
                DllImportError(e);
            }
            catch (BadImageFormatException e)
            {
                DllImportError(e);
            }
        }

        /// <summary>
        /// DllImportの際にDllNotFoundExceptionかBadImageFormatExceptionが発生した際に呼び出されるメソッド。
        /// エラーメッセージを表示して解決策をユーザに示す。
        /// </summary>
        /// <param name="ex"></param>
        public static void DllImportError(Exception ex)
        {
            throw CreateException(ex);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ex"></param>
        public static TrtException CreateException(Exception ex)
        {
            if (ex is null)
                throw new ArgumentNullException(nameof(ex));
            return new TrtException(ex.Message, ex);
        }
    }

}
