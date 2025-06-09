#!/bin/bash
#GPU configurations
#SBATCH --job-name=nvidiaGPU_image_processing
#SBATCH --account=project_2013968 
#SBATCH --partition=gpusmall
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=output.txt


#initialize variables
FILE=""
MODULE=""
CHANNEL=""
INFO=""
#parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--file)
      FILE="$2"
      shift 2
      ;;
    -m|--module)
      MODULE="$2"
      shift 2
      ;;
    -c|--channel)
      CHANNEL="$2"
      shift 2
      ;;
    -i|--info)
      INFO="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 -f <file> -m <module> -c <channel> -i <info>"
      exit 1
      ;;
  esac
done

#check for required arguments
if [[ -z "$FILE" || -z "$MODULE" ]]; then
  echo "Missing required arguments."
  echo "Usage: $0 -f <file> -m <module> -c <channel>(optional) -k <kernel>(optional)"
  exit 1
fi
#build the command
CMD="srun build/GPU_app -f \"$FILE\" -m \"$MODULE\""
[ -n "$CHANNEL" ] && CMD="$CMD -c \"$CHANNEL\""
[ -n "$INFO" ] && CMD="$CMD -k \"$INFO\""
#run the application
echo "Running: $CMD"
eval $CMD