//source https://blog.demofox.org/2015/04/19/frequency-domain-audio-synthesis-with-ifft-and-oscillators/

#define _CRT_SECURE_NO_WARNINGS
   
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
   
#define _USE_MATH_DEFINES
#include 
   
//=====================================================================================
// SNumeric - uses phantom types to enforce type safety
//=====================================================================================
template
struct SNumeric
{
public:
    explicit SNumeric(const T &value) : m_value(value) { }
    SNumeric() : m_value() { }
    inline T& Value() { return m_value; }
    inline const T& Value() const { return m_value; }
   
    typedef SNumeric TType;
    typedef T TInnerType;
   
    // Math Operations
    TType operator+ (const TType &b) const
    {
        return TType(this->Value() + b.Value());
    }
   
    TType operator- (const TType &b) const
    {
        return TType(this->Value() - b.Value());
    }
   
    TType operator* (const TType &b) const
    {
        return TType(this->Value() * b.Value());
    }
   
    TType operator/ (const TType &b) const
    {
        return TType(this->Value() / b.Value());
    }
 
    TType operator% (const TType &b) const
    {
        return TType(this->Value() % b.Value());
    }
   
    TType& operator+= (const TType &b)
    {
        Value() += b.Value();
        return *this;
    }
   
    TType& operator-= (const TType &b)
    {
        Value() -= b.Value();
        return *this;
    }
   
    TType& operator*= (const TType &b)
    {
        Value() *= b.Value();
        return *this;
    }
   
    TType& operator/= (const TType &b)
    {
        Value() /= b.Value();
        return *this;
    }
   
    TType& operator++ ()
    {
        Value()++;
        return *this;
    }
   
    TType& operator-- ()
    {
        Value()--;
        return *this;
    }
   
    // Extended Math Operations
    template
    T Divide(const TType &b)
    {
        return ((T)this->Value()) / ((T)b.Value());
    }
   
    // Logic Operations
    bool operatorValue() < b.Value();
    }
    bool operatorValue()  (const TType &b) const {
        return this->Value() > b.Value();
    }
    bool operator>= (const TType &b) const {
        return this->Value() >= b.Value();
    }
    bool operator== (const TType &b) const {
        return this->Value() == b.Value();
    }
    bool operator!= (const TType &b) const {
        return this->Value() != b.Value();
    }
   
private:
    T m_value;
};
   
//=====================================================================================
// Typedefs
//=====================================================================================
   
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef int16_t int16;
typedef int32_t int32;
   
// type safe types!
typedef SNumeric        TFrequency;
typedef SNumeric           TFFTBin;
typedef SNumeric          TTimeMs;
typedef SNumeric            TTimeS;
typedef SNumeric         TSamples;
typedef SNumeric     TFractionalSamples;
typedef SNumeric         TDecibels;
typedef SNumeric        TAmplitude;
typedef SNumeric          TRadians;
typedef SNumeric          TDegrees;
   
//=====================================================================================
// Constants
//=====================================================================================
 
static const float c_pi = (float)M_PI;
static const float c_twoPi = c_pi * 2.0f;
 
//=====================================================================================
// Structs
//=====================================================================================
   
struct SSoundSettings
{
    TSamples        m_sampleRate;
    TSamples        m_sampleCount;
};
   
//=====================================================================================
// Conversion Functions
//=====================================================================================
 
inline TFFTBin FrequencyToFFTBin(TFrequency frequency, TFFTBin numBins, TSamples sampleRate)
{
    // bin = frequency * numBin / sampleRate
    return TFFTBin((uint32)(frequency.Value() * (float)numBins.Value() / (float)sampleRate.Value()));
}
 
inline TFrequency FFTBinToFrequency(TFFTBin bin, TFFTBin numBins, TSamples sampleRate)
{
    // frequency = bin * SampleRate / numBins
    return TFrequency((float)bin.Value() * (float)sampleRate.Value() / (float)numBins.Value());
}
 
inline TRadians DegreesToRadians(TDegrees degrees)
{
    return TRadians(degrees.Value() * c_pi / 180.0f);
}
 
inline TDegrees RadiansToDegrees(TRadians radians)
{
    return TDegrees(radians.Value() * 180.0f / c_pi);
}
 
inline TDecibels AmplitudeToDB(TAmplitude volume)
{
    return TDecibels(20.0f * log10(volume.Value()));
}
   
inline TAmplitude DBToAmplitude(TDecibels dB)
{
    return TAmplitude(pow(10.0f, dB.Value() / 20.0f));
}
 
TTimeS SamplesToSeconds(const SSoundSettings &s, TSamples samples)
{
    return TTimeS(samples.Divide(s.m_sampleRate));
}
   
TSamples SecondsToSamples(const SSoundSettings &s, TTimeS seconds)
{
    return TSamples((int)(seconds.Value() * (float)s.m_sampleRate.Value()));
}
   
TSamples MilliSecondsToSamples(const SSoundSettings &s, TTimeMs milliseconds)
{
    return SecondsToSamples(s, TTimeS((float)milliseconds.Value() / 1000.0f));
}
   
TTimeMs SecondsToMilliseconds(TTimeS seconds)
{
    return TTimeMs((uint32)(seconds.Value() * 1000.0f));
}
   
TFrequency Frequency(float octave, float note)
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
    return TFrequency((float)(440 * pow(2.0, ((double)((octave - 4) * 12 + note)) / 12.0)));
}
   
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
 
//=====================================================================================
// Audio Utils
//=====================================================================================
 
void EnvelopeSamples(std::vector& samples, TSamples envelopeTimeFrontBack)
{
    const TSamples c_frontEnvelopeEnd(envelopeTimeFrontBack);
    const TSamples c_backEnvelopeStart(samples.size() - envelopeTimeFrontBack.Value());
 
    for (TSamples index(0), numSamples(samples.size()); index < numSamples; ++index)
    {
        // calculate envelope
        TAmplitude envelope(1.0f);
        if (index < c_frontEnvelopeEnd)
            envelope = TAmplitude(index.Divide(envelopeTimeFrontBack));
        else if (index > c_backEnvelopeStart)
            envelope = TAmplitude(1.0f) - TAmplitude((index - c_backEnvelopeStart).Divide(envelopeTimeFrontBack));
 
        // apply envelope
        samples[index.Value()] *= envelope;
    }
}
   
void NormalizeSamples(std::vector& samples, TAmplitude maxAmplitude)
{
    // nothing to do if no samples
    if (samples.size() == 0)
        return;
   
    // 1) find the largest absolute value in the samples.
    TAmplitude largestAbsVal = TAmplitude(abs(samples.front().Value()));
    std::for_each(samples.begin() + 1, samples.end(), [&largestAbsVal](const TAmplitude &a)
        {
            TAmplitude absVal = TAmplitude(abs(a.Value()));
            if (absVal > largestAbsVal)
                largestAbsVal = absVal;
        }
    );
   
    // 2) adjust largestAbsVal so that when we divide all samples, none will be bigger than maxAmplitude
    // if the value we are going to divide by is <= 0, bail out
    largestAbsVal /= maxAmplitude;
    if (largestAbsVal <= TAmplitude(0.0f))
        return;
   
    // 3) divide all numbers by the largest absolute value seen so all samples are [-maxAmplitude,+maxAmplitude]
    for (TSamples index(0), numSamples(samples.size()); index < numSamples; ++index)
        samples[index.Value()] = samples[index.Value()] / largestAbsVal;
}
 
//=====================================================================================
// Wave File Writing Code
//=====================================================================================
struct SMinimalWaveFileHeader
{
    //the main chunk
    unsigned char m_szChunkID[4];      //0
    uint32        m_nChunkSize;        //4
    unsigned char m_szFormat[4];       //8
   
    //sub chunk 1 "fmt "
    unsigned char m_szSubChunk1ID[4];  //12
    uint32        m_nSubChunk1Size;    //16
    uint16        m_nAudioFormat;      //18
    uint16        m_nNumChannels;      //20
    uint32        m_nSampleRate;       //24
    uint32        m_nByteRate;         //28
    uint16        m_nBlockAlign;       //30
    uint16        m_nBitsPerSample;    //32
   
    //sub chunk 2 "data"
    unsigned char m_szSubChunk2ID[4];  //36
    uint32        m_nSubChunk2Size;    //40
   
    //then comes the data!
};
   
//this writes a wave file
template
bool WriteWaveFile(const char *fileName, const std::vector &samples, const SSoundSettings &sound)
{
    //open the file if we can
    FILE *file = fopen(fileName, "w+b");
    if (!file)
        return false;
   
    //calculate bits per sample and the data size
    const int32 bitsPerSample = sizeof(T) * 8;
    const int dataSize = samples.size() * sizeof(T);
   
    SMinimalWaveFileHeader waveHeader;
   
    //fill out the main chunk
    memcpy(waveHeader.m_szChunkID, "RIFF", 4);
    waveHeader.m_nChunkSize = dataSize + 36;
    memcpy(waveHeader.m_szFormat, "WAVE", 4);
   
    //fill out sub chunk 1 "fmt "
    memcpy(waveHeader.m_szSubChunk1ID, "fmt ", 4);
    waveHeader.m_nSubChunk1Size = 16;
    waveHeader.m_nAudioFormat = 1;
    waveHeader.m_nNumChannels = 1;
    waveHeader.m_nSampleRate = sound.m_sampleRate.Value();
    waveHeader.m_nByteRate = sound.m_sampleRate.Value() * 1 * bitsPerSample / 8;
    waveHeader.m_nBlockAlign = 1 * bitsPerSample / 8;
    waveHeader.m_nBitsPerSample = bitsPerSample;
   
    //fill out sub chunk 2 "data"
    memcpy(waveHeader.m_szSubChunk2ID, "data", 4);
    waveHeader.m_nSubChunk2Size = dataSize;
   
    //write the header
    fwrite(&waveHeader, sizeof(SMinimalWaveFileHeader), 1, file);
   
    //write the wave data itself, converting it from float to the type specified
    std::vector outSamples;
    outSamples.resize(samples.size());
    for (size_t index = 0; index < samples.size(); ++index)
        outSamples[index] = AmplitudeToAudioSample(samples[index]);
    fwrite(&outSamples[0], dataSize, 1, file);
   
    //close the file and return success
    fclose(file);
    return true;
}
 
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
 
//=====================================================================================
// Wave forms
//=====================================================================================
 
void SineWave(CArray &frequencies, TFFTBin bin, TRadians startingPhase)
{
    // set up the single harmonic
    frequencies[bin.Value()] = std::polar(1.0f, startingPhase.Value());
}
 
void SawWave(CArray &frequencies, TFFTBin bin, TRadians startingPhase)
{
    // set up each harmonic
    const float volumeAdjustment = 2.0f / c_pi;
    const size_t bucketWalk = bin.Value();
    for (size_t harmonic = 1, bucket = bin.Value(); bucket < frequencies.size() / 2; ++harmonic, bucket += bucketWalk)
        frequencies[bucket] = std::polar(volumeAdjustment / (float)harmonic, startingPhase.Value());
}
 
void SquareWave(CArray &frequencies, TFFTBin bin, TRadians startingPhase)
{
    // set up each harmonic
    const float volumeAdjustment = 4.0f / c_pi;
    const size_t bucketWalk = bin.Value() * 2;
    for (size_t harmonic = 1, bucket = bin.Value(); bucket < frequencies.size() / 2; harmonic += 2, bucket += bucketWalk)
        frequencies[bucket] = std::polar(volumeAdjustment / (float)harmonic, startingPhase.Value());
}
 
void TriangleWave(CArray &frequencies, TFFTBin bin, TRadians startingPhase)
{
    // set up each harmonic
    const float volumeAdjustment = 8.0f / (c_pi*c_pi);
    const size_t bucketWalk = bin.Value() * 2;
    for (size_t harmonic = 1, bucket = bin.Value(); bucket < frequencies.size() / 2; harmonic += 2, bucket += bucketWalk, startingPhase *= TRadians(-1.0f))
        frequencies[bucket] = std::polar(volumeAdjustment / ((float)harmonic*(float)harmonic), startingPhase.Value());
}
 
void NoiseWave(CArray &frequencies, TFFTBin bin, TRadians startingPhase)
{
    // give a random amplitude and phase to each frequency
    for (size_t bucket = 0; bucket < frequencies.size() / 2; ++bucket)
    {
        float amplitude = static_cast  (rand()) / static_cast  (RAND_MAX);
        float phase = 2.0f * c_pi * static_cast  (rand()) / static_cast  (RAND_MAX);
        frequencies[bucket] = std::polar(amplitude, phase);
    }
}
 
//=====================================================================================
// Tests
//=====================================================================================
 
template
void ConsantBins(
    const W &waveForm,
    TFrequency &frequency,
    bool repeat,
    const char *fileName,
    bool normalize,
    TAmplitude multiplier,
    TRadians startingPhase = DegreesToRadians(TDegrees(270.0f))
)
{
    const TFFTBin c_numBins(4096);
 
    //our desired sound parameters
    SSoundSettings sound;
    sound.m_sampleRate = TSamples(44100);
    sound.m_sampleCount = MilliSecondsToSamples(sound, TTimeMs(500));
 
    // allocate space for the output file and initialize it
    std::vector samples;
    samples.resize(sound.m_sampleCount.Value());
    std::fill(samples.begin(), samples.end(), TAmplitude(0.0f));
 
    // make test data
    CArray data(c_numBins.Value());
    waveForm(data, FrequencyToFFTBin(frequency, c_numBins, sound.m_sampleRate), startingPhase);
 
    // inverse fft - convert from frequency domain (frequencies) to time domain (samples)
    // need to scale up amplitude before fft
    data *= (float)data.size();
    ifft(data);
 
    // convert to samples
    if (repeat)
    {
        //repeat results in the output buffer
        size_t dataSize = data.size();
        for (size_t i = 0; i < samples.size(); ++i)
            samples[i] = TAmplitude((float)data[i%dataSize].real());
    }
    else
    {
        //put results in the output buffer once.  Useful for debugging / visualization
        for (size_t i = 0; i < samples.size() && i < data.size(); ++i)
            samples[i] = TAmplitude((float)data[i].real());
    }
 
    // normalize our samples if we should
    if (normalize)
        NormalizeSamples(samples, DBToAmplitude(TDecibels(-3.0f)));
 
    // apply the multiplier passed in
    std::for_each(samples.begin(), samples.end(), [&](TAmplitude& amplitude) {
        amplitude *= multiplier;
    });
 
    // write the wave file
    WriteWaveFile(fileName, samples, sound);
}
 
void Convolve_Circular(const std::vector& a, const std::vector& b, std::vector& result)
{
    // NOTE: Written for readability, not efficiency
    TSamples ASize(a.size());
    TSamples BSize(b.size());
 
    // NOTE: the circular convolution result doesn't have to be the size of a, i just chose this size to match the ifft
    // circular convolution output.
    result.resize(ASize.Value());
    std::fill(result.begin(), result.end(), TAmplitude(0.0f));
 
    for (TSamples outputSampleIndex(0), numOutputSamples(ASize); outputSampleIndex < numOutputSamples; ++outputSampleIndex)
    {
        TAmplitude &outputSample = result[outputSampleIndex.Value()];
        for (TSamples sampleIndex(0), numSamples(ASize); sampleIndex < numSamples; ++sampleIndex)
        {
            TSamples BIndex = (outputSampleIndex + ASize - sampleIndex) % ASize;
            if (BIndex < BSize)
            {
                const TAmplitude &ASample = a[sampleIndex.Value()];
                const TAmplitude &BSample = b[BIndex.Value()];
                outputSample += BSample * ASample;
            }
        }
    }
}
 
void Convolve_Linear(const std::vector& a, const std::vector& b, std::vector& result)
{
    // NOTE: Written for readability, not efficiency
    TSamples ASize(a.size());
    TSamples BSize(b.size());
 
    result.resize(ASize.Value() + BSize.Value() - 1);
    std::fill(result.begin(), result.end(), TAmplitude(0.0f));
 
    for (TSamples outputSampleIndex(0), numOutputSamples(result.size()); outputSampleIndex < numOutputSamples; ++outputSampleIndex)
    {
        TAmplitude &outputSample = result[outputSampleIndex.Value()];
        for (TSamples sampleIndex(0), numSamples(ASize); sampleIndex = sampleIndex)
            {
                TSamples BIndex = outputSampleIndex - sampleIndex;
                if (BIndex < BSize)
                {
                    const TAmplitude &ASample = a[sampleIndex.Value()];
                    const TAmplitude &BSample = b[BIndex.Value()];
                    outputSample += BSample * ASample;
                }
            }
        }
    }
}
 
 
template
void DoConvolution(const W1 &waveForm1, const W2 &waveForm2)
{
    const TFFTBin c_numBins(4096);
 
    //our desired sound parameters
    SSoundSettings sound;
    sound.m_sampleRate = TSamples(44100);
    sound.m_sampleCount = TSamples(c_numBins.Value());
 
    // make the frequency data for wave form 1
    CArray data1(c_numBins.Value());
    waveForm1(data1, TFFTBin(1), DegreesToRadians(TDegrees(270.0f)));
 
    // make the frequency data for wave form 2
    CArray data2(c_numBins.Value());
    waveForm2(data2, TFFTBin(1), DegreesToRadians(TDegrees(270.0f)));
 
    // do circular convolution in time domain by doing multiplication in the frequency domain
    CArray data3(c_numBins.Value());
    data3 = data1 * data2;
 
    // write out the first convolution input (in time domain samples)
    std::vector samples1;
    samples1.resize(sound.m_sampleCount.Value());
    std::fill(samples1.begin(), samples1.end(), TAmplitude(0.0f));
    {
        data1 *= (float)data1.size();
        ifft(data1);
 
        // convert to samples
        for (size_t i = 0; i < samples1.size() && i < data1.size(); ++i)
            samples1[i] = TAmplitude((float)data1[i].real());
 
        // write the wave file
        WriteWaveFile("_convolution_A.wav", samples1, sound);
    }
 
    // write out the second convolution input (in time domain samples)
    std::vector samples2;
    samples2.resize(sound.m_sampleCount.Value());
    std::fill(samples2.begin(), samples2.end(), TAmplitude(0.0f));
    {
        data2 *= (float)data2.size();
        ifft(data2);
 
        // convert to samples
        for (size_t i = 0; i < samples2.size() && i < data2.size(); ++i)
            samples2[i] = TAmplitude((float)data2[i].real());
 
        // write the wave file
        WriteWaveFile("_convolution_B.wav", samples2, sound);
    }
 
    // write the result of the convolution (in time domain samples)
    {
        data3 *= (float)data3.size();
        ifft(data3);
 
        // convert to samples
        std::vector samples3;
        samples3.resize(sound.m_sampleCount.Value());
        for (size_t i = 0; i < samples3.size() && i < data3.size(); ++i)
            samples3[i] = TAmplitude((float)data3[i].real());
 
        // write the wave file
        NormalizeSamples(samples3, TAmplitude(1.0f));
        WriteWaveFile("_convolution_out_ifft.wav", samples3, sound);
    }
 
    // do linear convolution in the time domain and write out the wave file
    {
        std::vector samples4;
        Convolve_Linear(samples1, samples2, samples4);
        NormalizeSamples(samples4, TAmplitude(1.0f));
        WriteWaveFile("_convolution_out_lin.wav", samples4, sound);
    }
 
    // do circular convolution in time domain and write out the wave file
    {
        std::vector samples4;
        Convolve_Circular(samples1, samples2, samples4);
        NormalizeSamples(samples4, TAmplitude(1.0f));
        WriteWaveFile("_convolution_out_cir.wav", samples4, sound);
    }
}
 
//=====================================================================================
// Frequency Over Time Track Structs
//=====================================================================================
 
struct SBinTrack
{
    SBinTrack() { }
 
    SBinTrack(
        TFFTBin bin,
        std::function amplitudeFunction,
        TRadians phase = DegreesToRadians(TDegrees(270.0f))
    )
        : m_bin(bin)
        , m_amplitudeFunction(amplitudeFunction)
        , m_phase(phase) {}
 
    TFFTBin                                     m_bin;
    std::function   m_amplitudeFunction;
    TRadians                                    m_phase;
};
 
//=====================================================================================
// Frequency Amplitude Over Time Test
//=====================================================================================
 
void TracksToSamples_IFFT_Window(const std::vector &tracks, CArray &windowData, TTimeS time, TTimeS totalTime)
{
    // clear out the window data
    windowData = Complex(0.0f, 0.0f);
 
    // gather the bin data
    std::for_each(tracks.begin(), tracks.end(), [&](const SBinTrack &track) {
        windowData[track.m_bin.Value()] = std::polar(track.m_amplitudeFunction(time, totalTime).Value(), track.m_phase.Value());
    });
 
    // convert it to time samples
    windowData *= (float)windowData.size();
    ifft(windowData);
}
 
void TracksToSamples_IFFT(const std::vector &tracks, std::vector &samples, TTimeS totalTime, TFFTBin numBins)
{
    // convert the tracks to samples, one window of numBins at a time
    CArray windowData(numBins.Value());
    for (TSamples startSample(0), numSamples(samples.size()); startSample < numSamples; startSample += TSamples(numBins.Value()))
    {
        // Convert the tracks that we can into time samples using ifft
        float percent = startSample.Divide(numSamples);
        TTimeS time(totalTime.Value() * percent);
        TracksToSamples_IFFT_Window(tracks, windowData, time, totalTime);
 
        // convert window data to samples
        const size_t numWindowSamples = std::min(numBins.Value(), (numSamples - startSample).Value());
        for (size_t i = 0; i < numWindowSamples; ++i)
            samples[startSample.Value() + i] = TAmplitude((float)windowData[i].real());
    }
}
 
void TracksToSamples_Oscilators(const std::vector &tracks, std::vector &samples, TTimeS totalTime, TFFTBin numBins)
{
    // Render each time/amplitude track in each frequency bin to actual cosine samples
    float samplesPerSecond = (float)samples.size() / totalTime.Value();
    float ratio = samplesPerSecond / (float)numBins.Value();
    for (size_t i = 0, c = samples.size(); i < c; ++i)
    {
        float percent = (float)i / (float)c;
        TTimeS time(totalTime.Value() * percent);
        samples[i].Value() = 0.0f;
        std::for_each(tracks.begin(), tracks.end(),
            [&](const SBinTrack &track)
            {
                TAmplitude amplitude = track.m_amplitudeFunction(time, totalTime);
                samples[i] += TAmplitude(cos(time.Value()*c_twoPi*ratio*(float)track.m_bin.Value() + track.m_phase.Value())) * amplitude;
            }
        );
    }
}
 
struct SFadePair
{
    TTimeS m_time;
    TAmplitude m_amplitude;
};
 
std::function MakeFadeFunction(std::initializer_list fadePairs)
{
    // if no faid pairs, 0 amplitude always
    if (fadePairs.size() == 0)
    {
        return [](TTimeS time, TTimeS totalTime) -> TAmplitude
        {
            return TAmplitude(0.0f);
        };
    }
 
    // otherwise, use the fade info to make an amplitude over time track
    // NOTE: assume amplitude 0 at time 0 and totalTime
    return [fadePairs](TTimeS time, TTimeS totalTime) -> TAmplitude
    {
        TTimeS lastFadeTime(0.0f);
        TAmplitude lastFadeAmplitude(0.0f);
 
        for (size_t i = 0; i < fadePairs.size(); ++i)
        {
            if (time < fadePairs.begin()[i].m_time)
            {
                TAmplitude percent(((time - lastFadeTime) / (fadePairs.begin()[i].m_time - lastFadeTime)).Value());
                return percent * (fadePairs.begin()[i].m_amplitude - lastFadeAmplitude) + lastFadeAmplitude;
            }
            lastFadeTime = fadePairs.begin()[i].m_time;
            lastFadeAmplitude = fadePairs.begin()[i].m_amplitude;
        }
        if (time < totalTime)
        {
            TAmplitude percent(((time - lastFadeTime) / (totalTime - lastFadeTime)).Value());
            return percent * (TAmplitude(0.0f) - lastFadeAmplitude) + lastFadeAmplitude;
        }
 
        return TAmplitude(0.0f);
    };
}
 
void DynamicBins(TFFTBin numBins, const std::vector& tracks, const char *fileNameFFT, const char * fileNameOsc)
{
    //our desired sound parameters
    SSoundSettings sound;
    sound.m_sampleRate = TSamples(44100);
    sound.m_sampleCount = MilliSecondsToSamples(sound, TTimeMs(2000));
 
    // allocate space for the output file and initialize it
    std::vector samples;
    samples.resize(sound.m_sampleCount.Value());
    std::fill(samples.begin(), samples.end(), TAmplitude(0.0f));
 
    const TTimeS totalTime = SamplesToSeconds(sound, sound.m_sampleCount);
 
    // convert our frequency over time descriptions to time domain samples using IFFT
    if (fileNameFFT)
    {
        TracksToSamples_IFFT(tracks, samples, totalTime, numBins);
        NormalizeSamples(samples, DBToAmplitude(TDecibels(-3.0f)));
        EnvelopeSamples(samples, MilliSecondsToSamples(sound, TTimeMs(50)));
        WriteWaveFile(fileNameFFT, samples, sound);
    }
 
    // convert our frequency over time descriptions to time domain samples using Oscillators
    // and additive synthesis
    if (fileNameOsc)
    {
        TracksToSamples_Oscilators(tracks, samples, totalTime, numBins);
        NormalizeSamples(samples, DBToAmplitude(TDecibels(-3.0f)));
        EnvelopeSamples(samples, MilliSecondsToSamples(sound, TTimeMs(50)));
        WriteWaveFile(fileNameOsc, samples, sound);
    }
}
 
//=====================================================================================
// Main
//=====================================================================================
int main(int argc, char **argv)
{
     // make some basic wave forms with IFFT
    ConsantBins(NoiseWave, Frequency(3, 8), true, "_noise.wav", true, TAmplitude(1.0f));
    ConsantBins(SquareWave, Frequency(3, 8), true, "_square.wav", true, TAmplitude(1.0f));
    ConsantBins(TriangleWave, Frequency(3, 8), true, "_triangle.wav", true, TAmplitude(1.0f));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw.wav", true, TAmplitude(1.0f));
    ConsantBins(SineWave, Frequency(3, 8), true, "_cosine.wav", true, TAmplitude(1.0f), TRadians(0.0f));
    ConsantBins(SineWave, Frequency(3, 8), true, "_sine.wav", true, TAmplitude(1.0f));
 
    // show saw wave phase shifted.  Looks different but sounds the same!
    // You can do the same with square, saw, triangle (and other more complex wave forms)
    // We take the saw waves down 12 db though because some variations have large peaks so would clip otherwise.
    // we don't normalize because we want you to hear them all at the same loudness to tell that they really do sound the same.
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_0.wav"  , false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(0.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_15.wav" , false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(15.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_30.wav" , false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(30.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_45.wav" , false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(45.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_60.wav" , false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(60.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_75.wav" , false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(75.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_90.wav" , false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(90.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_105.wav", false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(105.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_120.wav", false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(120.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_135.wav", false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(135.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_150.wav", false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(150.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_165.wav", false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(165.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_180.wav", false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(180.0f)));
    ConsantBins(SawWave, Frequency(3, 8), true, "_saw_270.wav", false, DBToAmplitude(TDecibels(-12.0f)), DegreesToRadians(TDegrees(270.0f)));
 
    // show how IFFT can have popping at window edges
    {
        std::vector tracks;
        tracks.emplace_back(SBinTrack(TFFTBin(10), [](TTimeS time, TTimeS totalTime) -> TAmplitude
        {
            return TAmplitude(cos(time.Value()*c_twoPi*4.0f) * 0.5f + 0.5f);
        }));
 
        // make a version that starts at a phase of 0 degrees and has popping at the
        // edges of each IFFT window
        tracks.front().m_phase = TRadians(0.0f);
        DynamicBins(TFFTBin(1024), tracks, "_IFFTTest1.wav", nullptr);
 
        // make a version that starts at a phase of 270 degrees and is smooth at the
        // edges of each IFFT window but can only change amplitude at the edges of
        // each window.
        tracks.front().m_phase = TRadians(DegreesToRadians(TDegrees(270.0f)));
        DynamicBins(TFFTBin(1024), tracks, "_IFFTTest2.wav", nullptr);
 
        // make a version with oscillators and additive synthesis which has no
        // popping and can also change amplitude anywhere in the wave form.
        DynamicBins(TFFTBin(1024), tracks, nullptr, "_IFFTTest3.wav");
    }
 
    // make an alien sound using both IFFT and oscillators (additive synthesis)
    {
        std::vector tracks;
        tracks.emplace_back(SBinTrack(TFFTBin(1), MakeFadeFunction({ { TTimeS(0.5f), TAmplitude(1.0f) }, { TTimeS(1.0f), TAmplitude(0.5f) } })));
        tracks.emplace_back(SBinTrack(TFFTBin(2), MakeFadeFunction({ { TTimeS(1.0f), TAmplitude(1.0f) } })));
        tracks.emplace_back(SBinTrack(TFFTBin(3), MakeFadeFunction({ { TTimeS(1.5f), TAmplitude(1.0f) } })));
        tracks.emplace_back(SBinTrack(TFFTBin(5), MakeFadeFunction({ { TTimeS(1.25f), TAmplitude(1.0f) } })));
        tracks.emplace_back(SBinTrack(TFFTBin(10), [](TTimeS time, TTimeS totalTime) -> TAmplitude
        {
            float value = (cos(time.Value()*c_twoPi*4.0f) * 0.5f + 0.5f) * 0.5f;
            if (time < totalTime * TTimeS(0.5f))
                value *= (time / (totalTime*TTimeS(0.5f))).Value();
            else
                value *= 1.0f - ((time - totalTime*TTimeS(0.5f)) / (totalTime*TTimeS(0.5f))).Value();
            return TAmplitude(value);
        }));
        DynamicBins(TFFTBin(1024), tracks, "_alien_ifft.wav", "_alien_osc.wav");
    }
 
    // Make some drum beats
    {
        // frequency = bin * SampleRate / numBins
        // frequency = bin * 44100 / 4096
        // frequency ~= bin * 10.75
        // up to 22050hz at bin 2048
 
        TFFTBin c_numBins(4096);
        TSamples c_sampleRate(44100);
 
        std::vector tracks;
 
        const TTimeS timeMultiplier(1.1f);
 
        // base drum: 100-200hz every half second
        {
            const TFFTBin start(FrequencyToFFTBin(TFrequency(100.0f), c_numBins, c_sampleRate));
            const TFFTBin end(FrequencyToFFTBin(TFrequency(200.0f), c_numBins, c_sampleRate));
            const TFFTBin step(5);
 
            auto beat = [&](TTimeS time, TTimeS totalTime)->TAmplitude
            {
                time *= timeMultiplier;
                time = TTimeS(std::fmod(time.Value(), 0.5f));
                const TTimeS attack(0.01f);
                const TTimeS release(TTimeS(0.2f) - attack);
                const TTimeS totalBeatTime(attack + release);
 
                TAmplitude ret;
                if (time < attack)
                    ret = TAmplitude(time.Divide(attack));
                else if (time < totalBeatTime)
                    ret = TAmplitude(1.0f) - TAmplitude((time - attack).Divide(release));
                else
                    ret = TAmplitude(0.0f);
                return ret * TAmplitude(10.0f);
            };
            for (TFFTBin i = start; i TAmplitude
            {
                time *= timeMultiplier;
                time = TTimeS(std::fmod(time.Value() + 0.25f, 1.0f));
                const TTimeS attack(0.025f);
                const TTimeS release(TTimeS(0.075f) - attack);
                const TTimeS totalBeatTime(attack + release);
 
                TAmplitude ret;
                if (time < attack)
                    ret = TAmplitude(time.Divide(attack));
                else if (time < totalBeatTime)
                    ret = TAmplitude(1.0f) - TAmplitude((time - attack).Divide(release));
                else
                    ret = TAmplitude(0.0f);
                return ret;
            };
            for (TFFTBin i = start; i TAmplitude
            {
                time *= timeMultiplier;
                time = TTimeS(std::fmod(time.Value() + 0.75f, 1.0f));
                const TTimeS attack(0.025f);
                const TTimeS release(TTimeS(0.075f) - attack);
                const TTimeS totalBeatTime(attack + release);
 
                TAmplitude ret;
                if (time < attack)
                    ret = TAmplitude(time.Divide(attack));
                else if (time < totalBeatTime)
                    ret = TAmplitude(1.0f) - TAmplitude((time - attack).Divide(release));
                else
                    ret = TAmplitude(0.0f);
                return ret;
            };
            for (TFFTBin i = start; i <= end; i += step)
                tracks.emplace_back(SBinTrack(i, beat));
        }
 
        // render the result with both ifft and oscillators
        DynamicBins(c_numBins, tracks, "_drums_ifft.wav", "_drums_osc.wav");
    }
 
    // do our convolution tests
    DoConvolution(SawWave, SquareWave);
 
    return 0;
}