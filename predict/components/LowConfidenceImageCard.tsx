import Image from 'next/image';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';

interface Detection {
    class: string;
    confidence: number;
    bbox: [number, number, number, number];
}

interface LowConfImage {
    filename: string;
    max_confidence: number;
    detections: Detection[];
}

interface LowConfidenceImageCardProps {
    image: LowConfImage;
    onClick: () => void;
}

export function LowConfidenceImageCard({ image, onClick }: LowConfidenceImageCardProps) {
    return (
        <Card className="overflow-hidden cursor-pointer" onClick={onClick}>
            <CardHeader className="pb-2">
                <CardTitle className="text-lg truncate" title={image.filename}>
                    {image.filename}
                </CardTitle>
                <CardDescription>
                    Max Confidence: {(image.max_confidence * 100).toFixed(1)}%
                </CardDescription>
            </CardHeader>
            <CardContent className="p-0">
                <div className="aspect-square relative bg-gray-100 dark:bg-gray-800">
                    <Image
                        src={`http://localhost:5000/images/${image.filename}`}
                        alt={image.filename}
                        fill
                        className="object-cover"
                        sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                    />
                    {image.detections.length > 0 && (
                        <svg
                            className="absolute inset-0 w-full h-full pointer-events-none"
                            viewBox="0 0 100 100"
                            preserveAspectRatio="none"
                        >
                            {image.detections.map((detection, index) => {
                                const [x_center, y_center, width, height] = detection.bbox;
                                const x = (x_center - width / 2) * 100;
                                const y = (y_center - height / 2) * 100;
                                const w = width * 100;
                                const h = height * 100;
                                return (
                                    <rect
                                        key={index}
                                        x={x}
                                        y={y}
                                        width={w}
                                        height={h}
                                        fill="none"
                                        stroke="red"
                                        strokeWidth="0.5"
                                    />
                                );
                            })}
                        </svg>
                    )}
                </div>
                {image.detections.length > 0 && (
                    <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 h-24 overflow-y-auto">
                        <p className="text-sm font-medium text-yellow-700 dark:text-yellow-400 mb-2">
                            Detections:
                        </p>
                        <ul className="space-y-1">
                            {image.detections.map((detection, index) => (
                                <li key={index} className="text-xs text-yellow-600 dark:text-yellow-300">
                                    {detection.class}: {(detection.confidence * 100).toFixed(1)}%
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
                {image.detections.length === 0 && (
                    <div className="p-4 bg-gray-50 dark:bg-gray-800">
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                            No detections found
                        </p>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
