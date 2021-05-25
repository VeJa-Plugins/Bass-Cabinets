#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "fftw3.h"
#include "lv2/lv2plug.in/ns/lv2core/lv2.h"
#include "sndfile.h"

/**********************************************************************************************************************************************************/
#define REAL 0
#define IMAG 1

#define SAMPLERATE 48000

#define SIZE (512 * 4)

//plugin URI
#define PLUGIN_URI "http://VeJaPlugins.com/plugins/Release/FreqGen"

//macro for Volume in DB to a coefficient
#define DB_CO(g) ((g) > -90.0f ? powf(10.0f, (g) * 0.05f) : 0.0f)

typedef enum {IN, OUT}PortIndex;

typedef struct{
    float const *in;
    float *out;
    float *model;
    float *mode;
    float *outbuf;
    float *inbuf;
    float *IR;
    float *overlap;
    float *oA;
    float *oB;
    float *oC;
    int multiplier;
    int prev_model;
    int prev_mode;
    const float *attenuation;

    fftwf_complex *outComplex;
    fftwf_complex *IRout;
    fftwf_complex *convolved;

    fftwf_plan fft;
    fftwf_plan ifft;
    fftwf_plan IRfft;
} Cabsim;
/**********************************************************************************************************************************************************/
//functions

/**********************************************************************************************************************************************************/
static LV2_Handle
instantiate(const LV2_Descriptor*   descriptor,
double                              samplerate,
const char*                         bundle_path,
const LV2_Feature* const* features)
{
    Cabsim* cabsim = (Cabsim*)malloc(sizeof(Cabsim));

    cabsim->outComplex = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*(SIZE));
    cabsim->IRout =  (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*(SIZE));
    cabsim->convolved =  (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex)*(SIZE));

    cabsim->overlap = (float *) calloc((SIZE),sizeof(float));
    cabsim->outbuf = (float *) calloc((SIZE),sizeof(float));
    cabsim->inbuf = (float *) calloc((SIZE),sizeof(float));
    cabsim->IR = (float *) calloc((SIZE),sizeof(float));
    cabsim->oA = (float *) calloc((SIZE),sizeof(float));
    cabsim->oB = (float *) calloc((SIZE),sizeof(float));
    cabsim->oC = (float *) calloc((SIZE),sizeof(float));

    const char* wisdomFile = "freq-gen.wisdom";
    //open file A
    const size_t path_len    = strlen(bundle_path);
    const size_t file_len    = strlen(wisdomFile);
    const size_t len         = path_len + file_len;
    char*        wisdom_path = (char*)malloc(len + 1);
    snprintf(wisdom_path, len + 1, "%s%s", bundle_path, wisdomFile);

    if (fftwf_import_wisdom_from_filename(wisdom_path) != 0) {
        cabsim->ifft = fftwf_plan_dft_c2r_1d(SIZE, cabsim->convolved, cabsim->outbuf, FFTW_WISDOM_ONLY|FFTW_ESTIMATE);
    } 
    else {
        cabsim->ifft = fftwf_plan_dft_c2r_1d(SIZE, cabsim->convolved, cabsim->outbuf, FFTW_ESTIMATE);
    }

    cabsim->multiplier = 2;
    cabsim->prev_model = 999;
    cabsim->prev_mode = 999;

    free(wisdom_path);

    return (LV2_Handle)cabsim;
}
/**********************************************************************************************************************************************************/
static void connect_port(LV2_Handle instance, uint32_t port, void *data)
{
    Cabsim* cabsim = (Cabsim*)instance;

    switch ((PortIndex)port)
    {
        case IN:
            cabsim->in = (float*) data;
            break;
        case OUT:
            cabsim->out = (float*) data;
            break;
    }
}
/**********************************************************************************************************************************************************/
void activate(LV2_Handle instance)
{
}

/**********************************************************************************************************************************************************/
void run(LV2_Handle instance, uint32_t n_samples)
{
    Cabsim* cabsim = (Cabsim*)instance;    

    const float *in = cabsim->in;
    float *out = cabsim->out;
    float *outbuf = cabsim->outbuf;
    float *inbuf = cabsim->inbuf;
    float *IR = cabsim->IR;
    float *overlap = cabsim->overlap;
    float *oA = cabsim->oA;
    float *oB = cabsim->oB;
    float *oC = cabsim->oC;

    uint32_t i, j, m;

    if(n_samples == 128)
    {
        cabsim->multiplier = 16;
    }
    else if (n_samples == 256)
    {
        cabsim->multiplier = 8;
    }

    //set frequencys
    for (m = 0; m < ((n_samples / 2) * cabsim->multiplier) ;m++)
    {
        if (440.0f)
            cabsim->convolved[m][REAL] = 1.0f;
        else
            cabsim->convolved[m][REAL] = 0.0f;

        cabsim->convolved[m][IMAG] = 0.0f;
    }

    fftwf_execute(cabsim->ifft);

    //normalize output with overlap add.
    if(n_samples == 256)
    {
        for ( j = 0; j < n_samples * cabsim->multiplier; j++)
        {
            if(j < n_samples)
            {
                out[j] = ((outbuf[j] / (n_samples * cabsim->multiplier)) + overlap[j]);
            }
            else
            {
                overlap[j - n_samples] = outbuf[j]  / (n_samples * cabsim->multiplier);
            }
        }
    }
    else if (n_samples == 128)      //HIER VERDER GAAN!!!!!! oA, oB, oC changed malloc to calloc. (initiate buffer with all zeroes)
    {
        for ( j = 0; j < n_samples * cabsim->multiplier; j++)
        {
            if(j < n_samples)   //runs 128 times filling the output buffer with overap add
            {
                out[j] = (outbuf[j] / (n_samples * cabsim->multiplier) + oA[j] + oB[j] + oC[j]);
            }
            else
            {
                oC[j - n_samples] = oB[j]; // 128 samples of usefull data
                oB[j - n_samples] = oA[j];  //filled with samples 128 to 255 of usefull data
                oA[j - n_samples] = (outbuf[j] / (n_samples * cabsim->multiplier)); //filled with 384 samples
            }
        }
    }


/*_______________________________________________________________________________

    uint32_t i, j, m;

    if(n_samples == 128)
    {
        cabsim->multiplier = 16;
    }
    else if (n_samples == 256)
    {
        cabsim->multiplier = 8;
    }

    //copy inputbuffer and IR buffer with zero padding.
    if(cabsim->prev_model  != model)
    {
        for ( i = 0; i < n_samples * cabsim->multiplier; i++)
        {
            inbuf[i] = (i < n_samples) ? (in[i] * coef * 0.2f): 0.0f;
            IR[i] = (i < n_samples) ? convolK(model,i) : 0.0f;
        }

        cabsim->prev_model = model;

        fftwf_execute(cabsim->IRfft);
    }
    else
    {
        for ( i = 0; i < n_samples * cabsim->multiplier; i++)
        {
            inbuf[i] = (i < n_samples) ? (in[i] * coef * 0.2f): 0.0f;
        }
    }

    fftwf_execute(cabsim->fft);

    //complex multiplication
    for(m = 0; m < ((n_samples / 2) * cabsim->multiplier) ;m++)
    {
        //real component
        cabsim->convolved[m][REAL] = cabsim->outComplex[m][REAL] * cabsim->IRout[m][REAL] - cabsim->outComplex[m][IMAG] * cabsim->IRout[m][IMAG];
        //imaginary component
        cabsim->convolved[m][IMAG] = cabsim->outComplex[m][REAL] * cabsim->IRout[m][IMAG] + cabsim->outComplex[m][IMAG] * cabsim->IRout[m][REAL];
    }

    fftwf_execute(cabsim->ifft);

    //normalize output with overlap add.
    if(n_samples == 256)
    {
        for ( j = 0; j < n_samples * cabsim->multiplier; j++)
        {
            if(j < n_samples)
            {
                out[j] = ((outbuf[j] / (n_samples * cabsim->multiplier)) + overlap[j]);
            }
            else
            {
                overlap[j - n_samples] = outbuf[j]  / (n_samples * cabsim->multiplier);
            }
        }
    }
    else if (n_samples == 128)      //HIER VERDER GAAN!!!!!! oA, oB, oC changed malloc to calloc. (initiate buffer with all zeroes)
    {
        for ( j = 0; j < n_samples * cabsim->multiplier; j++)
        {
            if(j < n_samples)   //runs 128 times filling the output buffer with overap add
            {
                out[j] = (outbuf[j] / (n_samples * cabsim->multiplier) + oA[j] + oB[j] + oC[j]);
            }
            else
            {
                oC[j - n_samples] = oB[j]; // 128 samples of usefull data
                oB[j - n_samples] = oA[j];  //filled with samples 128 to 255 of usefull data
                oA[j - n_samples] = (outbuf[j] / (n_samples * cabsim->multiplier)); //filled with 384 samples
            }
        }
    }*/
}

/**********************************************************************************************************************************************************/
void deactivate(LV2_Handle instance)
{
    // TODO: include the deactivate function code here
}
/**********************************************************************************************************************************************************/
void cleanup(LV2_Handle instance)
{
    Cabsim* cabsim = (Cabsim*)instance;
    fftwf_destroy_plan(cabsim->fft);
    fftwf_destroy_plan(cabsim->ifft);
    fftwf_destroy_plan(cabsim->IRfft);
    //free fft memory
    fftwf_free(cabsim->outComplex);
    fftwf_free(cabsim->IRout);
    fftwf_free(cabsim->convolved);
    //free allocated memory
    free(instance);
}
/**********************************************************************************************************************************************************/
const void* extension_data(const char* uri)
{
    return NULL;
}
/**********************************************************************************************************************************************************/
static const LV2_Descriptor Descriptor = {
    PLUGIN_URI,
    instantiate,
    connect_port,
    activate,
    run,
    deactivate,
    cleanup,
    extension_data
};
/**********************************************************************************************************************************************************/
LV2_SYMBOL_EXPORT
const LV2_Descriptor* lv2_descriptor(uint32_t index)
{
    if (index == 0) return &Descriptor;
    else return NULL;
}
/**********************************************************************************************************************************************************/
