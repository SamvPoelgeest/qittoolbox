import traceback

_LOGLEVELS = {'debug':0, 'min':1, 'info':2, 'warning':3, 'error':4, 'fatal':5}
_LOGLEVEL = 'debug'
_LOGFILE = None

def log(msg: str, msglevel: str) -> None:
    """
    Logs a message to console with the corresponding msglevel intensity. 

    INPUT:
        msg: str, the message to be displayed
        msglevel: str, the level of serverity given to the message by the function
    """
    global _LOGLEVEL, _LOGLEVELS, _LOGFILE

    if not msglevel in _LOGLEVELS:
        strings = [f'[Error] : [{__name__}] : [log] : Cannot log the following message, because an incorrect loglevel was provided.',\
                    f'==> Provided: msglevel={msglevel}. Printing original message anyway:',\
                    f'[{msglevel}] : {msg}' ]
        print('\n'.join(strings))

    elif _LOGLEVELS[msglevel] >= _LOGLEVELS[_LOGLEVEL]:
        print(f'[{msglevel}] : {msg}')

    log_to_file([msg],msglevel)

def log_to_file(strings: 'list[str]', msglevel: str) -> None:
    """
    Logs a message to file with the corresponding msglevel intensity.

    INPUT:
        strings: list of strings, the message to be logged.
        msglevel: str, the level of severity given to the message by the function.
    """
    global _LOGLEVEL, _LOGLEVELS, _LOGFILE
    if _LOGFILE is None:
        return

    if not msglevel in _LOGLEVELS:
        try:
            with open(_LOGFILE,'a') as f:
                f.write(f'[Error] : [{__name__}] : [log] : Cannot log the following message, because an incorrect loglevel was provided.\n')
                f.write(f'==> Provided: msglevel={msglevel}. Printing original message anyway:\n')
                f.write('\n'.join(strings) + '\n')
                f.close()
        except Exception as e:
            print(f'[Error] : [log_to_file] : Cannot log to file, error {e}, traceback: ')
            print(traceback.format_exc())
    
    elif _LOGLEVELS[msglevel] >= _LOGLEVELS[_LOGLEVEL]:
        try:
            with open(_LOGFILE,'a') as f:
                strings[0] = f'[{msglevel}] : ' + strings[0]
                f.write(f'\n[{msglevel}] : '.join(strings) + '\n')
                f.close()
        except Exception as e:
            print(f'[Error] : [log_to_file] : Cannot log to file, error {e}, traceback: ')
            print(traceback.format_exc())



def set_loglevel(loglevel: str) -> None:
    global _LOGLEVEL
    _LOGLEVEL = loglevel
    log(f'[set_loglevel] : set loglevel to `{loglevel}`', 'info')

def set_logfile(abs_path: str) -> None:
    global _LOGFILE
    _LOGFILE = abs_path
    log(f'[set_logfile] : set logfile to `{abs_path}`','info')