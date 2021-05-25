//for the original inspiration and C++ implementation of this code, please visit:
//source https://blog.demofox.org/2015/04/19/frequency-domain-audio-synthesis-with-ifft-and-oscillators/

//code addaptation by: Jan Janssen 24/05/21

/*
************************************************************************************************************************
*           INCLUDE FILES
************************************************************************************************************************
*/

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "utils.h"

/*
************************************************************************************************************************
*           LOCAL DEFINES
************************************************************************************************************************
*/


/*
************************************************************************************************************************
*           LOCAL CONSTANTS
************************************************************************************************************************
*/

static const float c_pi = (float)M_PI;
static const float c_twoPi = c_pi * 2.0f;

/*
************************************************************************************************************************
*           LOCAL DATA TYPES
************************************************************************************************************************
*/


/*
************************************************************************************************************************
*           LOCAL MACROS
************************************************************************************************************************
*/

/*
************************************************************************************************************************
*           LOCAL GLOBAL VARIABLES
************************************************************************************************************************
*/

/*
************************************************************************************************************************
*           LOCAL FUNCTION PROTOTYPES
************************************************************************************************************************
*/

uint32_t FrequencyToFFTBin(float frequency, uint32_t numBins, uint32_t sampleRate)
{
    // bin = frequency * numBin / sampleRate
    return (uint32_t)(frequency * (float)numBins / (float)sampleRate);
}
 
float FFTBinToFrequency(uint32_t bin, uint32_t numBins, uint32_t sampleRate)
{
    // frequency = bin * SampleRate / numBins
    return bin * (float)sampleRate / (float)numBins;
}
 
float DegreesToRadians(float degrees)
{
    return (degrees * c_pi / 180.0f);
}
 
float RadiansToDegrees(float radians)
{
    return (radians * 180.0f / c_pi);
}
 
float AmplitudeToDB(float volume)
{
    return (20.0f * log10(volume));
}
   
float DBToAmplitude(float dB)
{
    return (pow(10.0f, dB / 20.0f));
}
 
float SamplesToSeconds(uint32_t sample_rate, uint32_t samples)
{
    return (float)samples / (float)sample_rate;
}
   
uint32_t SecondsToSamples(uint32_t sample_rate, float seconds)
{
    return (uint32_t)(seconds * m_sampleRate);
}
   
uint32_t MilliSecondsToSamples(uint32_t sample_rate, float milliseconds)
{
    return SecondsToSamples(sample_rate, (milliseconds / 1000.0f));
}
   
float SecondsToMilliseconds(float seconds)
{
    return (seconds * 1000.0f);
}
   
float Frequency(float octave, float note)
{
    /* frequency = 440×(2^(n/12))
    Notes:
    0  = A
    1  = A#
    2  = B
    3  = C
    4  = C#
    5  = D
    6  = D#
    7  = E
    8  = F
    9  = F#
    10 = G
    11 = G# */
    return ((float)(440 * pow(2.0, ((double)((octave - 4) * 12 + note)) / 12.0)));
}

//TODO CHECK IF THIS IS ENEDED? OR WE CAN JUST LIMIT BETWEEN -1 and 1 for LV2?
/*
template
T AmplitudeToAudioSample(const TAmplitude& in)
{
    const T c_min = std::numeric_limits::min();
    const T c_max = std::numeric_limits::max();
    const float c_minFloat = (float)c_min;
    const float c_maxFloat = (float)c_max;
   
    float ret = in.Value() * c_maxFloat;
   
    if (ret  c_maxFloat)
        return c_max;
   
    return (T)ret;
}
*/


//=====================================================================================
// Audio Utils
//=====================================================================================
 

// envelopeTimeFrontBack:
/*
     ENV TIME
   ____________
  /            \
 /              \
/                \

*/
void EnvelopeSamples(float *samples, uint32_t frame_size, float envelopeTimeFrontBack)
{
	// nothing to do if no samples
    if (!samples)
        return;

    const uint32_t c_frontEnvelopeEnd = envelopeTimeFrontBack;
    const uint32_t c_backEnvelopeStart = frame_size - envelopeTimeFrontBack;
 
    for (uint32_t index = 0, uint32_t numSamples = frame_size; index < numSamples; ++index)
    {
        // calculate envelope
        float envelope = 1.0f;
        if (index < c_frontEnvelopeEnd)
            envelope = index / envelopeTimeFrontBack;
        else if (index > c_backEnvelopeStart)
            envelope = (1.0f) - (((index - c_backEnvelopeStart) / (envelopeTimeFrontBack)));
 
        // apply envelope
        samples[index] *= envelope;
    }
}
   
void NormalizeSamples(float *samples, uint32_t frame_size, float maxAmplitude)
{
    // nothing to do if no samples
    if (!samples)
        return;
   
    // 1) find the largest absolute value in the samples.
    float largestAbsVal = (abs(samples[0]));

    for (uint32_t index = 0, uint32_t numSamples = frame_size; index < numSamples; ++index)
    {
            float absVal = abs(samples[index]);
            if (absVal > largestAbsVal)
                largestAbsVal = absVal;
    }
   
    // 2) adjust largestAbsVal so that when we divide all samples, none will be bigger than maxAmplitude
    // if the value we are going to divide by is <= 0, bail out
    largestAbsVal /= maxAmplitude;
    if (largestAbsVal <= 0.0f)
        return;
   
    // 3) divide all numbers by the largest absolute value seen so all samples are [-maxAmplitude,+maxAmplitude]
    for (uint32_t index = 0, uint32_t numSamples = frame_size; index < numSamples; ++index)
        samples[index] = samples[index] / largestAbsVal;
}

//from the original source, we probably want to use fftw or similar.
/*
//=====================================================================================
// FFT / IFFT
//=====================================================================================
 
// Thanks RosettaCode.org!
// http://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
// In production you'd probably want a non recursive algorithm, but this works fine for us
 
// for use with FFT and IFFT
typedef std::complex Complex;
typedef std::valarray CArray;
 
// Cooley–Tukey FFT (in-place)
void fft(CArray& x)
{
    const size_t N = x.size();
    if (N <= 1) return;
  
    // divide
    CArray even = x[std::slice(0, N/2, 2)];
    CArray  odd = x[std::slice(1, N/2, 2)];
  
    // conquer
    fft(even);
    fft(odd);
  
    // combine
    for (size_t k = 0; k < N/2; ++k)
    {
        Complex t = std::polar(1.0f, -2 * c_pi * k / N) * odd[k];
        x[k    ] = even[k] + t;
        x[k+N/2] = even[k] - t;
    }
}
  
// inverse fft (in-place)
void ifft(CArray& x)
{
    // conjugate the complex numbers
    x = x.apply(std::conj);
  
    // forward fft
    fft( x );
  
    // conjugate the complex numbers again
    x = x.apply(std::conj);
  
    // scale the numbers
    x /= (float)x.size();
}
*/

//=====================================================================================
// Wave forms
//=====================================================================================
 
void SineWave(float *frequencies, uint32_t bin, float startingPhase)
{
    // set up the single harmonic
    frequencies[bin] = polar(1.0f, startingPhase);
}
 
void SawWave(float *frequencies, uint32_t frame_size, uint32_t bin, float startingPhase)
{
    // set up each harmonic
    const float volumeAdjustment = 2.0f / c_pi;
    const uint32_t bucketWalk = bin;
    for (uint32_t harmonic = 1, bucket = bin; bucket < frame_size / 2; ++harmonic, bucket += bucketWalk)
        frequencies[bucket] = polar(volumeAdjustment / (float)harmonic, startingPhase);
}
 
void SquareWave(float *frequencies, uint32_t frame_size, uint32_t bin, float startingPhase)
{
    // set up each harmonic
    const float volumeAdjustment = 4.0f / c_pi;
    const uint32_t bucketWalk = bin * 2;
    for (uint32_t harmonic = 1, bucket = bin; bucket < frame_size / 2; harmonic += 2, bucket += bucketWalk)
        frequencies[bucket] = polar(volumeAdjustment / (float)harmonic, startingPhase);
}
 
void TriangleWave(float *frequencies, uint32_t frame_size, uint32_t bin, float startingPhase)
{
    // set up each harmonic
    const float volumeAdjustment = 8.0f / (c_pi*c_pi);
    const uint32_t bucketWalk = bin * 2;
    for (uint32_t harmonic = 1, bucket = bin; bucket < frame_size / 2; harmonic += 2, bucket += bucketWalk, startingPhase *= -1.0f)
        frequencies[bucket] = polar(volumeAdjustment / ((float)harmonic*(float)harmonic), startingPhase);
}
 
void NoiseWave(float *frequencies, uint32_t frame_size, uint32_t bin, float startingPhase)
{
    // give a random amplitude and phase to each frequency
    for (uint32_t bucket = 0; bucket < frame_size / 2; ++bucket)
    {
        float amplitude = (rand()) / (RAND_MAX);
        float phase = 2.0f * c_pi * (rand()) / (RAND_MAX);
        frequencies[bucket] = polar(amplitude, phase);
    }
}

/*
************************************************************************************************************************
*           LOCAL CONFIGURATION ERRORS
************************************************************************************************************************
*/

/*
************************************************************************************************************************
*           LOCAL FUNCTIONS
************************************************************************************************************************
*/

/*
************************************************************************************************************************
*           GLOBAL FUNCTIONS
************************************************************************************************************************
*/