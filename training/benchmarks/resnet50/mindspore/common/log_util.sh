#!/bin/bash
MODE_NAME="Ais-Bench-Stubs"

date_format="+%Y-%m-%dT%T"

# Some useful colors.
# check if stdout is a terminal and support colors...
if [ -t 1 ] && [ "1$(tput colors 2>/dev/null)" -ge 18 ]; then
  readonly color_red="$(tput setaf 1)"
  readonly color_yellow="$(tput setaf 3)"
  readonly color_green="$(tput setaf 2)"
  readonly color_norm="$(tput sgr0)"
else 
  readonly color_red=""
  readonly color_yellow=""
  readonly color_green=""
  readonly color_norm=""
fi

if command -v caller >/dev/null 2>&1; then
  # return func(lineno:filename)
  # NOTE: skip 2-level inner frame
  _caller() { caller 2| awk '{sub(/.*\//,e,$3);print $2"("$3":"$1") "}'; }
else
  _caller() { :; }
fi

_log() 
{
  level=$1
  shift 1
  echo "$(date ${date_format}) -${MODE_NAME}- ${level} $(_caller)- $*"
}


logger_Debug() 
{
  echo "Debug $(_caller): $@"
}

logger_Info() 
{
  _log INFO "$@"
}

logger_Warn() 
{
  _log WARN "${color_yellow}$*${color_norm}"
}

logger_Error() 
{
  _log ERROR "${color_red}$*${color_norm}"
}

die() 
{
  _log ERROR "${color_red}$*${color_norm}"
  exit 1
}