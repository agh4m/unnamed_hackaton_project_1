'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle, ImageIcon, RefreshCw, Home } from 'lucide-react';
import { LowConfidenceImageCard } from '@/components/LowConfidenceImageCard';

interface LowConfImage {
    filename: string;
    max_confidence: number;
    detections: Array<{
        class: string;
        confidence: number;
        bbox: [number, number, number, number];
    }>;
}

export default function LowConfidencePage() {
    const router = useRouter();
    const [images, setImages] = useState<LowConfImage[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [refreshing, setRefreshing] = useState(false);

    useEffect(() => {
        fetchLowConfidenceImages();
    }, []);

    const fetchLowConfidenceImages = async (forceRefresh = false) => {
        try {
            setError(null);
            if (forceRefresh) setRefreshing(true);
            else setLoading(true);

            const url = forceRefresh
                ? 'http://localhost:5000/discover_low_confidence?refresh=true'
                : 'http://localhost:5000/discover_low_confidence';

            const response = await fetch(url);
            if (!response.ok) {
                throw new Error('Failed to fetch low confidence images');
            }
            const data = await response.json();
            const sortedImages = (data.low_confidence_images || []).sort(
                (a: LowConfImage, b: LowConfImage) => b.max_confidence - a.max_confidence
            );
            setImages(sortedImages);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-4 md:p-8">
                <div className="max-w-6xl mx-auto">
                    <Card>
                        <CardContent className="p-8 text-center">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                            <p className="text-gray-600 dark:text-gray-400">Scanning images for low confidence detections...</p>
                        </CardContent>
                    </Card>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-4 md:p-8">
                <div className="max-w-6xl mx-auto">
                    <Card>
                        <CardContent className="p-8 text-center">
                            <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                            <p className="text-red-700 dark:text-red-400">{error}</p>
                        </CardContent>
                    </Card>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-4 md:p-8">
            <div className="max-w-6xl mx-auto">
                <Card className="mb-8">
                    <CardHeader className="text-center">
                        <CardTitle className="text-3xl font-bold text-gray-800 dark:text-gray-100 flex items-center justify-center gap-2">
                            <AlertTriangle className="w-8 h-8 text-yellow-500" />
                            Low Confidence Images
                        </CardTitle>
                        <CardDescription className="text-lg text-gray-600 dark:text-gray-400">
                            Images with uncertain YOLO detections (confidence &lt; 0.7)
                            {!loading && !error && (
                                <span className="block mt-2 font-semibold">
                                    {images.length} low confidence image{images.length !== 1 ? 's' : ''} found
                                </span>
                            )}
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="text-center space-y-4">
                        <div className="flex gap-4 justify-center">
                            <Link href="/">
                                <Button variant="outline" className="flex items-center gap-2">
                                    <Home className="w-4 h-4" />
                                    Back to Home
                                </Button>
                            </Link>
                            <Button
                                onClick={() => fetchLowConfidenceImages(true)}
                                disabled={loading || refreshing}
                                variant="outline"
                                className="flex items-center gap-2"
                            >
                                <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
                                {refreshing ? 'Regenerating...' : 'Force Regenerate'}
                            </Button>
                        </div>
                    </CardContent>
                </Card>

                {images.length === 0 ? (
                    <Card>
                        <CardContent className="p-8 text-center">
                            <ImageIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                            <p className="text-gray-600 dark:text-gray-400">No low confidence images found.</p>
                        </CardContent>
                    </Card>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {images.map((image) => (
                            <LowConfidenceImageCard
                                key={image.filename}
                                image={image}
                                onClick={() => router.push(`/edit/${image.filename}`)}
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
