'use client';

import { useState } from 'react';
import { useSearchParams } from 'next/navigation';
import Image from 'next/image';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Upload, ImageIcon, Clock, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

export default function Home() {
    const searchParams = useSearchParams();
    const modelParam = searchParams.get('model');
    const [file, setFile] = useState<File | null>(null);
    const [modelType] = useState<'custom' | 'yolo'>(modelParam === 'custom' ? 'custom' : 'yolo');
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const [results, setResults] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile) {
            setFile(selectedFile);
            setImageUrl(URL.createObjectURL(selectedFile));
            setResults(null);
        }
    };

    const handleSubmit = async () => {
        if (!file) return;

        setLoading(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('modelType', modelType);

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            setResults(data);
        } catch (error) {
            console.error(error);
            setResults({ error: 'Failed to predict' });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-4 md:p-8">
            <div className="max-w-4xl mx-auto">
                <Card className="mb-8">
                    <CardHeader className="text-center">
                        <CardTitle className="text-3xl font-bold text-gray-800 dark:text-gray-100">Drone Spectrogram Classifier</CardTitle>
                        <CardDescription className="text-lg text-gray-600 dark:text-gray-400">
                            Upload a spectrogram image to classify drone signals using AI
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="text-center">
                        <Link href="/low-confidence">
                            <Button variant="outline" className="flex items-center gap-2">
                                <AlertTriangle className="w-4 h-4" />
                                View Low Confidence Images
                            </Button>
                        </Link>
                    </CardContent>
                </Card>

                <div className="grid md:grid-cols-2 gap-8">
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <Upload className="w-5 h-5" />
                                Upload Image
                            </CardTitle>
                            <CardDescription>
                                Select a PNG spectrogram file to analyze
                            </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-6 text-center hover:border-gray-400 dark:hover:border-gray-500 transition-colors">
                                <input
                                    type="file"
                                    accept=".png"
                                    onChange={handleFileChange}
                                    className="hidden"
                                    id="file-upload"
                                />
                                <label htmlFor="file-upload" className="cursor-pointer">
                                    <ImageIcon className="w-12 h-12 mx-auto mb-4 text-gray-400 dark:text-gray-500" />
                                    <p className="text-gray-600 dark:text-gray-400">
                                        {file ? file.name : 'Click to upload or drag and drop'}
                                    </p>
                                </label>
                            </div>

                            {imageUrl && (
                                <div className="mt-4">
                                    <Image
                                        src={imageUrl}
                                        alt="Uploaded spectrogram"
                                        width={400}
                                        height={400}
                                        className="rounded-lg shadow-md mx-auto"
                                    />
                                </div>
                            )}

                            <Button
                                onClick={handleSubmit}
                                disabled={!file || loading}
                                className="w-full"
                                size="lg"
                            >
                                {loading ? (
                                    <>
                                        <Clock className="w-4 h-4 mr-2 animate-spin" />
                                        Predicting...
                                    </>
                                ) : (
                                    <>
                                        <CheckCircle className="w-4 h-4 mr-2" />
                                        Predict
                                    </>
                                )}
                            </Button>
                        </CardContent>
                    </Card>

                    {results && (
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    {results.error ? (
                                        <XCircle className="w-5 h-5 text-red-500" />
                                    ) : (
                                        <CheckCircle className="w-5 h-5 text-green-500" />
                                    )}
                                    Results
                                </CardTitle>
                                <CardDescription className="text-gray-600 dark:text-gray-400">
                                    Analysis completed in {results.processingTime} ms
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Real Class</p>
                                    <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{results.realClass}</p>
                                </div>

                                {results.error ? (
                                    <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-200 dark:border-red-800">
                                        <p className="text-red-700 dark:text-red-400">{results.error}</p>
                                    </div>
                                ) : modelType === 'custom' ? (
                                    <div className="space-y-4">
                                        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                                            <p className="text-sm font-medium text-blue-700 dark:text-blue-400">Predicted Class</p>
                                            <p className="text-lg font-semibold text-blue-900 dark:text-blue-100">{results.predicted_class}</p>
                                        </div>
                                        <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                                            <p className="text-sm font-medium text-green-700 dark:text-green-400">Confidence</p>
                                            <p className="text-lg font-semibold text-green-900 dark:text-green-100">{results.confidence?.toFixed(4)}</p>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                                        <p className="text-sm font-medium text-purple-700 dark:text-purple-400 mb-2">Detected Classes</p>
                                        <ul className="space-y-2">
                                            {results.predicted_classes?.map((item: any, index: number) => (
                                                <li key={index} className="flex justify-between items-center bg-white dark:bg-gray-700 p-2 rounded border dark:border-gray-600">
                                                    <span className="font-medium text-gray-900 dark:text-gray-100">{item.class}</span>
                                                    <span className="text-sm text-gray-600 dark:text-gray-400">
                                                        {item.confidence?.toFixed(4)}
                                                    </span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    )}
                </div>
            </div>
        </div>
    );
}
